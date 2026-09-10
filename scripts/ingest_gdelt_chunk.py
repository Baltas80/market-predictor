"""Download one bounded GDELT historical chunk with resumable daily checkpoints.

A source-day that is genuinely unavailable is recorded in a deterministic
manifest and the chunk continues. Every successfully normalized day is also
checkpointed immediately, so a later workflow run can restore completed days
from the previous run's artifact and skip their downloads. Missing days are
never imputed here; a later recovery pass may retry them.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from io import BytesIO
from pathlib import Path
import time
import zipfile

import pandas as pd
import requests

from market_predictor.data_sources import _gdelt_category
from market_predictor.research_schema import deduplicate_events, normalize_event_sources

GDELT_DAY_RETRIES = 4
GDELT_RETRY_BASE_SECONDS = 2

# GDELT 2.0 has used both the original 58-field event layout and the
# later 61-field layout with Actor1Geo_ADM2, Actor2Geo_ADM2 and
# ActionGeo_ADM2. Historical files must be decoded according to the
# layout actually present in each file; otherwise DATEADDED/SOURCEURL
# become misaligned and the PIT gate rejects the entire day.
GDELT_EXPORT_COLUMNS_61 = [
    "global_event_id", "sql_date", "month_year", "year", "fraction_date",
    "actor1_code", "actor1_name", "actor1_country", "actor1_known_group",
    "actor1_ethnic", "actor1_religion1", "actor1_religion2", "actor1_type1",
    "actor1_type2", "actor1_type3", "actor2_code", "actor2_name",
    "actor2_country", "actor2_known_group", "actor2_ethnic", "actor2_religion1",
    "actor2_religion2", "actor2_type1", "actor2_type2", "actor2_type3",
    "is_root_event", "event_code", "event_base_code", "event_root_code",
    "quad_class", "goldstein_scale", "num_mentions", "num_sources",
    "num_articles", "avg_tone", "actor1_geo_type", "actor1_geo_fullname",
    "actor1_geo_country", "actor1_geo_adm1", "actor1_geo_adm2",
    "actor1_geo_lat", "actor1_geo_long", "actor1_geo_feature_id",
    "actor2_geo_type", "actor2_geo_fullname", "actor2_geo_country",
    "actor2_geo_adm1", "actor2_geo_adm2", "actor2_geo_lat", "actor2_geo_long",
    "actor2_geo_feature_id", "action_geo_type", "action_geo_fullname",
    "action_geo_country", "action_geo_adm1", "action_geo_adm2",
    "action_geo_lat", "action_geo_long", "action_geo_feature_id",
    "date_added", "source_url",
]
GDELT_EXPORT_COLUMNS_58 = [
    column for column in GDELT_EXPORT_COLUMNS_61
    if column not in {"actor1_geo_adm2", "actor2_geo_adm2", "action_geo_adm2"}
]


def load_gdelt_day(day: str | pd.Timestamp) -> pd.DataFrame:
    """Load and normalize one GDELT 2.0 daily event export."""
    date = pd.Timestamp(day).strftime("%Y%m%d")
    url = f"https://data.gdeltproject.org/events/{date}.export.CSV.zip"
    response = requests.get(
        url,
        timeout=120,
        headers={"User-Agent": "market-predictor/0.1 (research ingestion)"},
    )
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        members = archive.namelist()
        if not members:
            raise ValueError(f"Empty GDELT archive for {date}")
        member = members[0]
        with archive.open(member) as probe:
            first_line = probe.readline()
        field_count = first_line.count(b"\t") + 1
        if field_count == len(GDELT_EXPORT_COLUMNS_61):
            columns = GDELT_EXPORT_COLUMNS_61
        elif field_count == len(GDELT_EXPORT_COLUMNS_58):
            columns = GDELT_EXPORT_COLUMNS_58
        else:
            raise ValueError(
                f"Unsupported GDELT field count for {date}: "
                f"{field_count} (expected 58 or 61)"
            )
        with archive.open(member) as handle:
            frame = pd.read_csv(
                handle,
                sep="\t",
                header=None,
                names=columns,
                dtype=str,
                low_memory=False,
            )

    for column in ("actor1_geo_adm2", "actor2_geo_adm2", "action_geo_adm2"):
        if column not in frame:
            frame[column] = pd.NA
    frame = frame[GDELT_EXPORT_COLUMNS_61]
    frame["date_added"] = pd.to_datetime(
        frame["date_added"], format="%Y%m%d%H%M%S", utc=True, errors="coerce"
    )
    frame["sql_date"] = pd.to_datetime(
        frame["sql_date"], format="%Y%m%d", errors="coerce", utc=True
    )
    for column in ["goldstein_scale", "num_mentions", "num_sources", "num_articles", "avg_tone"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    actor_text = frame[["actor1_name", "actor2_name", "action_geo_fullname"]].fillna("").agg(" ".join, axis=1)
    frame["category"] = [
        _gdelt_category(root, code, text)
        for root, code, text in zip(frame["event_root_code"], frame["event_code"], actor_text)
    ]
    scale = frame["goldstein_scale"].abs().clip(0, 10) / 10
    media = frame["num_sources"].fillna(0).clip(lower=0)
    frame["severity"] = (0.7 * scale + 0.3 * (media / (media + 5)).clip(0, 1)).clip(0, 1)
    frame["surprise"] = 0.0
    return frame


def _checkpoint_path(checkpoint_dir: Path, day: str) -> Path:
    return checkpoint_dir / f"day_{pd.Timestamp(day).date().isoformat()}.csv"


def _read_checkpoint(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("GDELT checkpoint is empty; redownloading")
    for column in ("event_time", "published_at", "available_at"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


def fetch_gdelt_chunk(start: str, end: str, checkpoint_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    if end_date < start_date:
        raise ValueError("GDELT end must be on or after start")

    checkpoint_dir = checkpoint_dir or Path("data/gdelt_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    missing: list[dict[str, str]] = []
    total_days = (end_date - start_date).days + 1
    current = start_date
    completed = 0
    restored = 0
    dropped_unavailable = 0

    while current <= end_date:
        checkpoint = _checkpoint_path(checkpoint_dir, current.isoformat())
        if checkpoint.exists():
            try:
                restored_frame = _read_checkpoint(checkpoint)
                frames.append(restored_frame)
                restored += 1
                completed += 1
                print(
                    f"GDELT checkpoint restored: {current.isoformat()} rows={len(restored_frame)} "
                    f"({completed}/{total_days})",
                    flush=True,
                )
                current += timedelta(days=1)
                continue
            except (OSError, pd.errors.ParserError, ValueError) as exc:
                print(
                    f"Invalid GDELT checkpoint for {current.isoformat()}: {type(exc).__name__}: {exc}; redownloading",
                    flush=True,
                )
                checkpoint.unlink(missing_ok=True)

        last_error: Exception | None = None
        for attempt in range(1, GDELT_DAY_RETRIES + 1):
            try:
                raw_day = load_gdelt_day(current)
                if "event_id" not in raw_day and "global_event_id" in raw_day:
                    raw_day = raw_day.rename(columns={"global_event_id": "event_id"})
                normalized = normalize_event_sources(raw_day, source_id="GDELT_2_Event_Database")
                before = len(normalized)
                normalized = normalized.dropna(subset=["event_id", "event_time", "available_at", "severity"]).copy()
                dropped_unavailable += before - len(normalized)
                normalized.to_csv(
                    checkpoint,
                    index=False,
                    lineterminator="\n",
                    date_format="%Y-%m-%dT%H:%M:%S%z",
                )
                frames.append(normalized)
                last_error = None
                print(
                    f"GDELT day checkpoint saved: {current.isoformat()} rows={len(normalized)}",
                    flush=True,
                )
                break
            except (requests.RequestException, TimeoutError, ValueError, zipfile.BadZipFile) as exc:
                last_error = exc
                if attempt >= GDELT_DAY_RETRIES:
                    break
                delay = GDELT_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                print(
                    f"GDELT retry {attempt}/{GDELT_DAY_RETRIES - 1} for {current.isoformat()} "
                    f"after {type(exc).__name__}: {exc}; waiting {delay}s",
                    flush=True,
                )
                time.sleep(delay)

        if last_error is not None:
            missing.append({
                "date": current.isoformat(),
                "source_id": "GDELT_2_Event_Database",
                "status": "missing",
                "error_type": type(last_error).__name__,
                "error": str(last_error),
            })
            print(
                f"GDELT gap recorded for {current.isoformat()} after {GDELT_DAY_RETRIES} attempts; continuing chunk",
                flush=True,
            )

        completed += 1
        if completed == 1 or completed % 25 == 0 or completed == total_days:
            print(
                f"GDELT staging progress: {completed}/{total_days} days ({completed / total_days:.1%}); "
                f"restored={restored}",
                flush=True,
            )
        current += timedelta(days=1)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result = deduplicate_events(combined) if not combined.empty else pd.DataFrame(columns=[
        "event_id", "event_time", "published_at", "available_at", "source_id",
        "category", "severity", "country", "entity", "sector", "duration_days",
        "media_intensity", "surprise",
    ])
    if dropped_unavailable:
        print(
            f"GDELT PIT filter: excluded {dropped_unavailable} rows without a valid event/availability/severity field",
            flush=True,
        )
    return result, pd.DataFrame(missing)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--missing-output", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output.parent / (output.stem + "_days")
    missing_output = Path(args.missing_output) if args.missing_output else output.with_name(output.stem + "_missing.csv")
    frame, missing = fetch_gdelt_chunk(args.start, args.end, checkpoint_dir=checkpoint_dir)
    frame.to_csv(output, index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z")
    missing.to_csv(missing_output, index=False, lineterminator="\n")
    print(
        f"GDELT chunk saved: {args.start} -> {args.end}; rows={len(frame)}; missing_days={len(missing)}; "
        f"restored/checkpointed={len(list(checkpoint_dir.glob('day_*.csv')))}; path={output}; missing_manifest={missing_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
