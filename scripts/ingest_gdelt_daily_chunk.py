"""Ingest GDELT 1.0 daily event files with conservative PIT availability.

The production historical lockbox currently uses the official GDELT 1.0 daily
stream at events/{YYYYMMDD}.export.CSV.zip. Those files are named for the
previous day's event-discovery date and are published the following morning.
We therefore never treat SQLDATE or the 8-digit file date as availability.
Instead, availability is assigned conservatively to 12:00 UTC on the day after
the file date. Publication time remains unknown.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
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
GDELT_CHECKPOINT_VERSION = "gdelt1-daily-pit-v2"
GDELT_SOURCE_ID = "GDELT_1_Daily_Event_Database"
GDELT_DAILY_URL = "https://data.gdeltproject.org/events/{date}.export.CSV.zip"

GDELT_COLUMNS_58 = [
    "global_event_id", "sql_date", "month_year", "year", "fraction_date",
    "actor1_code", "actor1_name", "actor1_country", "actor1_known_group",
    "actor1_ethnic", "actor1_religion1", "actor1_religion2", "actor1_type1",
    "actor1_type2", "actor1_type3", "actor2_code", "actor2_name",
    "actor2_country", "actor2_known_group", "actor2_ethnic", "actor2_religion1",
    "actor2_religion2", "actor2_type1", "actor2_type2", "actor2_type3",
    "is_root_event", "event_code", "event_base_code", "event_root_code",
    "quad_class", "goldstein_scale", "num_mentions", "num_sources",
    "num_articles", "avg_tone", "actor1_geo_type", "actor1_geo_fullname",
    "actor1_geo_country", "actor1_geo_adm1", "actor1_geo_lat", "actor1_geo_long",
    "actor1_geo_feature_id", "actor2_geo_type", "actor2_geo_fullname",
    "actor2_geo_country", "actor2_geo_adm1", "actor2_geo_lat", "actor2_geo_long",
    "actor2_geo_feature_id", "action_geo_type", "action_geo_fullname",
    "action_geo_country", "action_geo_adm1", "action_geo_lat", "action_geo_long",
    "action_geo_feature_id", "date_added", "source_url",
]


def _availability_for_file_date(file_date: date) -> pd.Timestamp:
    """Return a conservative availability boundary for a GDELT 1.0 file."""
    return pd.Timestamp(
        datetime.combine(file_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        + timedelta(hours=12)
    )


def load_gdelt_daily_day(day: str | pd.Timestamp) -> pd.DataFrame:
    file_date = pd.Timestamp(day).date()
    date_text = file_date.strftime("%Y%m%d")
    response = requests.get(
        GDELT_DAILY_URL.format(date=date_text),
        timeout=120,
        headers={"User-Agent": "market-predictor/0.1 (research ingestion)"},
    )
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        members = archive.namelist()
        if not members:
            raise ValueError(f"Empty GDELT daily archive for {date_text}")
        member = members[0]
        with archive.open(member) as probe:
            field_count = probe.readline().count(b"\t") + 1
        if field_count != len(GDELT_COLUMNS_58):
            raise ValueError(
                f"Unexpected GDELT 1.0 field count for {date_text}: {field_count}; expected 58"
            )
        with archive.open(member) as handle:
            frame = pd.read_csv(
                handle,
                sep="\t",
                header=None,
                names=GDELT_COLUMNS_58,
                dtype=str,
                low_memory=False,
            )

    frame["sql_date"] = pd.to_datetime(
        frame["sql_date"].astype("string").str.strip(),
        format="%Y%m%d",
        utc=True,
        errors="coerce",
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
    frame["event_time"] = frame["sql_date"]
    frame["published_at"] = pd.NaT
    frame["available_at"] = _availability_for_file_date(file_date)
    frame["event_id"] = frame["global_event_id"].astype("string")
    return normalize_event_sources(frame, source_id=GDELT_SOURCE_ID)


def _invalid_event_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify structurally unusable source rows before chunk-level validation."""
    data = frame.copy()
    text_invalid = pd.Series(False, index=data.index)
    for column in ("event_id", "source_id", "category"):
        values = data[column].astype("string").str.strip().str.lower()
        text_invalid |= data[column].isna() | values.isin({"", "nan", "none", "nat", "<na>"})
    timestamps_invalid = (
        data[["event_time", "available_at"]].isna().any(axis=1)
        | (data["available_at"] < data["event_time"])
    )
    severity = pd.to_numeric(data["severity"], errors="coerce")
    severity_invalid = severity.isna() | ~severity.map(pd.notna)
    return text_invalid | timestamps_invalid | severity_invalid


def _quarantine_invalid_rows(frame: pd.DataFrame, source_day: date, checkpoint_dir: Path) -> tuple[pd.DataFrame, int]:
    """Quarantine malformed rows with an auditable sidecar instead of poisoning the whole chunk."""
    invalid = _invalid_event_mask(frame)
    if not invalid.any():
        return frame, 0
    rejected = frame.loc[invalid].copy()
    rejected.insert(0, "source_day", source_day.isoformat())
    rejected.insert(1, "quarantine_reason", "invalid_required_event_field")
    path = checkpoint_dir / f"invalid_{source_day.isoformat()}.csv"
    rejected.to_csv(path, index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z")
    clean = frame.loc[~invalid].copy()
    print(
        f"GDELT 1.0 quarantined {len(rejected)} malformed rows for {source_day.isoformat()}; "
        f"sidecar={path}",
        flush=True,
    )
    return clean, len(rejected)


def _checkpoint_path(checkpoint_dir: Path, day: str) -> Path:
    return checkpoint_dir / f"day_{pd.Timestamp(day).date().isoformat()}.csv"


def _read_checkpoint(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty or "checkpoint_version" not in frame.columns:
        raise ValueError("stale or empty GDELT 1.0 checkpoint")
    if frame["checkpoint_version"].ne(GDELT_CHECKPOINT_VERSION).any():
        raise ValueError("incompatible GDELT 1.0 checkpoint version")
    frame = frame.drop(columns=["checkpoint_version"])
    for column in ("event_time", "published_at", "available_at"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    required = ["event_id", "event_time", "available_at", "severity"]
    if not set(required).issubset(frame.columns) or frame[required].isna().any().any():
        raise ValueError("invalid GDELT 1.0 checkpoint PIT fields")
    return frame


def fetch_gdelt_daily_chunk(start: str, end: str, checkpoint_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    if end_date < start_date:
        raise ValueError("end must be on or after start")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    missing: list[dict[str, str]] = []
    current = start_date
    total = (end_date - start_date).days + 1
    completed = 0

    while current <= end_date:
        checkpoint = _checkpoint_path(checkpoint_dir, current.isoformat())
        if checkpoint.exists():
            try:
                restored = _read_checkpoint(checkpoint)
                frames.append(restored)
                completed += 1
                current += timedelta(days=1)
                continue
            except (OSError, pd.errors.ParserError, ValueError):
                checkpoint.unlink(missing_ok=True)

        last_error: Exception | None = None
        for attempt in range(1, GDELT_DAY_RETRIES + 1):
            try:
                normalized = load_gdelt_daily_day(current)
                normalized, _ = _quarantine_invalid_rows(normalized, current, checkpoint_dir)
                if normalized.empty:
                    raise ValueError(f"GDELT 1.0 source-day {current.isoformat()} has no valid event rows")
                normalized["checkpoint_version"] = GDELT_CHECKPOINT_VERSION
                normalized.to_csv(checkpoint, index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z")
                frames.append(normalized.drop(columns=["checkpoint_version"]))
                last_error = None
                break
            except (requests.RequestException, TimeoutError, ValueError, zipfile.BadZipFile) as exc:
                last_error = exc
                if attempt < GDELT_DAY_RETRIES:
                    time.sleep(GDELT_RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
        if last_error is not None:
            missing.append({
                "date": current.isoformat(),
                "source_id": GDELT_SOURCE_ID,
                "status": "missing",
                "error_type": type(last_error).__name__,
                "error": str(last_error),
            })
        completed += 1
        if completed == 1 or completed % 25 == 0 or completed == total:
            print(f"GDELT 1.0 staging progress: {completed}/{total} days", flush=True)
        current += timedelta(days=1)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result = deduplicate_events(combined) if not combined.empty else pd.DataFrame(columns=[
        "event_id", "event_time", "published_at", "available_at", "source_id",
        "category", "severity", "country", "entity", "sector", "duration_days",
        "media_intensity", "surprise",
    ])
    return result, pd.DataFrame(missing)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--missing-output", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame, missing = fetch_gdelt_daily_chunk(args.start, args.end, Path(args.checkpoint_dir))
    frame.to_csv(output, index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z")
    missing.to_csv(args.missing_output, index=False, lineterminator="\n")
    print(f"GDELT 1.0 chunk saved: {args.start} -> {args.end}; rows={len(frame)}; missing_days={len(missing)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
