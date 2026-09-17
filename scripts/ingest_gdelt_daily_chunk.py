"""Compatibility wrapper for the canonical GDELT 1.0 daily ingestion path.

This legacy script remains available for historical workflows that call its
original function names, but all parsing and PIT semantics are delegated to
``market_predictor.gdelt1``. There is intentionally no second DATEADDED/PIT
implementation in this module.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import time

import pandas as pd
import requests

from market_predictor.gdelt1 import GDELT_SOURCE_ID, gdelt1_daily_availability, load_gdelt_day
from market_predictor.research_schema import deduplicate_events, normalize_event_sources

GDELT_DAY_RETRIES = 4
GDELT_RETRY_BASE_SECONDS = 2
GDELT_CHECKPOINT_VERSION = "gdelt1-daily-pit-v2"
GDELT_DAILY_URL = "https://data.gdeltproject.org/events/{date}.export.CSV.zip"


def _availability_for_file_date(file_date: date | str | pd.Timestamp) -> pd.Timestamp:
    """Delegate availability calculation to the canonical DST-aware PIT rule."""
    return gdelt1_daily_availability(file_date)


def load_gdelt_daily_day(day: str | pd.Timestamp) -> pd.DataFrame:
    """Compatibility adapter backed by the canonical GDELT 1.0 loader."""
    raw = load_gdelt_day(day).copy()
    if "event_id" not in raw.columns and "global_event_id" in raw.columns:
        raw = raw.rename(columns={"global_event_id": "event_id"})
    raw["event_time"] = raw["sql_date"]
    raw["published_at"] = pd.NaT
    if "available_at" not in raw.columns:
        raise ValueError("canonical GDELT 1.0 loader returned no explicit available_at")
    return normalize_event_sources(raw, source_id=GDELT_SOURCE_ID)


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
    if "source_id" in frame and frame["source_id"].astype(str).ne(GDELT_SOURCE_ID).any():
        raise ValueError("invalid GDELT 1.0 checkpoint source identifier")
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
                if normalized.empty:
                    raise ValueError(f"GDELT 1.0 source-day {current.isoformat()} returned no rows")
                normalized["checkpoint_version"] = GDELT_CHECKPOINT_VERSION
                normalized.to_csv(checkpoint, index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z")
                frames.append(normalized.drop(columns=["checkpoint_version"]))
                last_error = None
                break
            except (requests.RequestException, TimeoutError, ValueError) as exc:
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
