"""Ingest GDELT 2.0 15-minute event exports for a bounded date chunk.

Production research track: GDELT 2.0 is the primary event source. Availability
is taken strictly from DATEADDED (UTC), never from the event day (SQLDATE).
The raw ZIP is processed and discarded; only normalized daily checkpoints and
chunk output are persisted. Missing 15-minute exports are recorded explicitly
and are fail-closed at final staging.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import Path
import time
import zipfile

import pandas as pd
import requests

from market_predictor.data_sources import _gdelt_category
from market_predictor.research_schema import deduplicate_events, normalize_event_sources

RETRIES = 4
RETRY_BASE_SECONDS = 2
CHECKPOINT_VERSION = "gdelt2-15min-pit-v1"
SOURCE_ID = "GDELT_2_Event_Database"
MASTER_URL = "https://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
BASE_URL = "https://data.gdeltproject.org/gdeltv2/"

COLUMNS_61 = [
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


def _date_added(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.isdigit() and len(text) == 14:
        return text
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite() or number != number.to_integral_value():
        return None
    text = format(number.to_integral_value(), "f")
    return text if text.isdigit() and len(text) == 14 else None


def _parse_date_added(series: pd.Series) -> pd.Series:
    canonical = series.map(_date_added).astype("string")
    return pd.to_datetime(canonical, format="%Y%m%d%H%M%S", utc=True, errors="coerce")


def _get_master(start: date, end: date) -> dict[str, str]:
    response = requests.get(MASTER_URL, timeout=120, headers={"User-Agent": "market-predictor/0.1 (research ingestion)"})
    response.raise_for_status()
    urls: dict[str, str] = {}
    for line in response.text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        url = parts[-1]
        name = url.rsplit("/", 1)[-1]
        if not name.endswith(".export.CSV.zip"):
            continue
        stamp = name[:14]
        try:
            ts = datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if start <= ts.date() <= end:
            urls[stamp] = url
    return urls


def _download(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.get(url, timeout=180, headers={"User-Agent": "market-predictor/0.1 (research ingestion)"})
            response.raise_for_status()
            return response.content
        except (requests.RequestException, TimeoutError) as exc:
            last = exc
            if attempt < RETRIES:
                time.sleep(RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
    raise RuntimeError(f"GDELT 2 download failed after {RETRIES} attempts: {url}: {last}")


def _read_export(content: bytes, filename: str) -> pd.DataFrame:
    with zipfile.ZipFile(BytesIO(content)) as archive:
        members = archive.namelist()
        if not members:
            raise ValueError(f"Empty GDELT 2 archive: {filename}")
        member = members[0]
        with archive.open(member) as probe:
            count = probe.readline().count(b"\t") + 1
        if count != 61:
            raise ValueError(f"Unexpected GDELT 2 field count in {filename}: {count}; expected 61")
        with archive.open(member) as handle:
            return pd.read_csv(handle, sep="\t", header=None, names=COLUMNS_61, dtype=str, low_memory=False)


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame["date_added"] = _parse_date_added(frame["date_added"])
    frame["sql_date"] = pd.to_datetime(frame["sql_date"].astype("string").str.strip(), format="%Y%m%d", utc=True, errors="coerce")
    for column in ("goldstein_scale", "num_mentions", "num_sources", "num_articles", "avg_tone"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    actor_text = frame[["actor1_name", "actor2_name", "action_geo_fullname"]].fillna("").agg(" ".join, axis=1)
    frame["category"] = [_gdelt_category(root, code, text) for root, code, text in zip(frame["event_root_code"], frame["event_code"], actor_text)]
    scale = frame["goldstein_scale"].abs().clip(0, 10) / 10
    media = frame["num_sources"].fillna(0).clip(lower=0)
    frame["severity"] = (0.7 * scale + 0.3 * (media / (media + 5)).clip(0, 1)).clip(0, 1)
    frame["surprise"] = 0.0
    frame["event_id"] = frame["global_event_id"].astype("string")
    frame["event_time"] = frame["sql_date"]
    frame["published_at"] = pd.NaT
    frame["available_at"] = frame["date_added"]
    normalized = normalize_event_sources(frame, source_id=SOURCE_ID)
    normalized = normalized.dropna(subset=["event_id", "event_time", "available_at", "severity"]).copy()
    normalized = normalized[normalized["available_at"] >= normalized["event_time"]].copy()
    return normalized


def fetch_chunk(start: str, end: str, checkpoint_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_date, end_date = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    if end_date < start_date:
        raise ValueError("end must be on or after start")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    master = _get_master(start_date, end_date)
    frames: list[pd.DataFrame] = []
    missing: list[dict[str, str]] = []
    current = start_date
    while current <= end_date:
        checkpoint = checkpoint_dir / f"day_{current.isoformat()}.csv"
        if checkpoint.exists():
            try:
                restored = pd.read_csv(checkpoint, parse_dates=["event_time", "published_at", "available_at"])
                if "checkpoint_version" in restored and restored["checkpoint_version"].eq(CHECKPOINT_VERSION).all():
                    frames.append(restored.drop(columns=["checkpoint_version"]))
                    current += timedelta(days=1)
                    continue
            except (OSError, pd.errors.ParserError, ValueError):
                pass
            checkpoint.unlink(missing_ok=True)

        day_frames: list[pd.DataFrame] = []
        for minute in range(0, 24 * 60, 15):
            stamp = datetime.combine(current, datetime.min.time(), tzinfo=timezone.utc) + timedelta(minutes=minute)
            key = stamp.strftime("%Y%m%d%H%M%S")
            url = master.get(key)
            if url is None:
                missing.append({"date": key, "source_id": SOURCE_ID, "status": "missing", "error_type": "MasterListGap", "error": "No English GDELT 2 export listed for expected 15-minute slot"})
                continue
            try:
                normalized = _normalize(_read_export(_download(url), url.rsplit("/", 1)[-1]))
                if not normalized.empty:
                    day_frames.append(normalized)
            except (requests.RequestException, RuntimeError, ValueError, zipfile.BadZipFile, pd.errors.ParserError) as exc:
                missing.append({"date": key, "source_id": SOURCE_ID, "status": "missing", "error_type": type(exc).__name__, "error": str(exc)})
        day = deduplicate_events(pd.concat(day_frames, ignore_index=True)) if day_frames else pd.DataFrame(columns=["event_id", "event_time", "published_at", "available_at", "source_id", "category", "severity", "country", "entity", "sector", "duration_days", "media_intensity", "surprise"])
        if day.empty:
            raise RuntimeError(f"GDELT 2 source-day {current.isoformat()} has no valid event rows")
        checkpoint_frame = day.copy()
        checkpoint_frame["checkpoint_version"] = CHECKPOINT_VERSION
        checkpoint_frame.to_csv(checkpoint, index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z")
        frames.append(day)
        print(f"GDELT 2.0 day complete: {current.isoformat()} rows={len(day)}", flush=True)
        current += timedelta(days=1)

    combined = deduplicate_events(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()
    return combined, pd.DataFrame(missing)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--missing-output", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    frame, missing = fetch_chunk(args.start, args.end, Path(args.checkpoint_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z")
    missing.to_csv(args.missing_output, index=False, lineterminator="\n")
    print(f"GDELT 2.0 chunk saved: {args.start} -> {args.end}; rows={len(frame)}; missing_15min_files={len(missing)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
