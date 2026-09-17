"""Download bounded GDELT 1.0 historical chunks with resumable checkpoints.

The source parser and PIT boundary are canonicalized in
``market_predictor.gdelt1``. This script owns only chunking, retries,
checkpointing and gap reporting.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import time
import zipfile

import pandas as pd
import requests

from market_predictor.gdelt1 import GDELT_SOURCE_ID, gdelt1_daily_availability, load_gdelt_day
from market_predictor.research_schema import deduplicate_events, normalize_event_sources

GDELT_DAY_RETRIES = 4
GDELT_RETRY_BASE_SECONDS = 2
GDELT_CHECKPOINT_VERSION = "gdelt1-pit-v4"


def _checkpoint_path(checkpoint_dir: Path, day: str) -> Path:
    return checkpoint_dir / f"day_{pd.Timestamp(day).date().isoformat()}.csv"


def _read_checkpoint(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("GDELT checkpoint is empty; redownloading")
    if "checkpoint_version" not in frame.columns or frame["checkpoint_version"].ne(GDELT_CHECKPOINT_VERSION).any():
        raise ValueError("GDELT checkpoint schema/version is stale; redownloading")
    frame = frame.drop(columns=["checkpoint_version"])
    for column in ("event_time", "published_at", "available_at"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    required = ["event_id", "event_time", "available_at", "severity"]
    if not set(required).issubset(frame.columns):
        raise ValueError("GDELT checkpoint lacks required PIT columns; redownloading")
    if frame[required].isna().any().any():
        raise ValueError("GDELT checkpoint contains invalid PIT fields; redownloading")
    if "source_id" in frame and frame["source_id"].ne(GDELT_SOURCE_ID).any():
        raise ValueError("GDELT checkpoint contains a non-canonical source identifier; redownloading")
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
                print(f"GDELT checkpoint restored: {current.isoformat()} rows={len(restored_frame)} ({completed}/{total_days})", flush=True)
                current += timedelta(days=1)
                continue
            except (OSError, pd.errors.ParserError, ValueError) as exc:
                print(f"Invalid GDELT checkpoint for {current.isoformat()}: {type(exc).__name__}: {exc}; redownloading", flush=True)
                checkpoint.unlink(missing_ok=True)

        last_error: Exception | None = None
        for attempt in range(1, GDELT_DAY_RETRIES + 1):
            try:
                raw_day = load_gdelt_day(current)
                if raw_day.empty:
                    raise ValueError(f"GDELT source-day {current.isoformat()} returned an empty export")
                raw_day = raw_day.rename(columns={"global_event_id": "event_id"}) if "event_id" not in raw_day else raw_day
                raw_day["event_time"] = raw_day["sql_date"]
                raw_day["published_at"] = pd.NaT
                if "available_at" not in raw_day:
                    raise ValueError("canonical GDELT 1.0 loader returned no explicit available_at")
                normalized = normalize_event_sources(raw_day, source_id=GDELT_SOURCE_ID)
                before = len(normalized)
                invalid_counts = {column: int(normalized[column].isna().sum()) for column in ("event_id", "event_time", "available_at", "severity")}
                normalized = normalized.dropna(subset=["event_id", "event_time", "available_at", "severity"]).copy()
                dropped_unavailable += before - len(normalized)
                if before and len(normalized) == 0:
                    raise ValueError("GDELT PIT normalization rejected every row; " + ", ".join(f"{column}_nulls={count}" for column, count in invalid_counts.items()))
                normalized["checkpoint_version"] = GDELT_CHECKPOINT_VERSION
                normalized.to_csv(checkpoint, index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z")
                normalized = normalized.drop(columns=["checkpoint_version"])
                frames.append(normalized)
                last_error = None
                print(f"GDELT day checkpoint saved: {current.isoformat()} rows={len(normalized)} available_at={gdelt1_daily_availability(current).isoformat()}", flush=True)
                break
            except (requests.RequestException, TimeoutError, ValueError, zipfile.BadZipFile) as exc:
                last_error = exc
                if attempt >= GDELT_DAY_RETRIES:
                    break
                delay = GDELT_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                print(f"GDELT retry {attempt}/{GDELT_DAY_RETRIES - 1} for {current.isoformat()} after {type(exc).__name__}: {exc}; waiting {delay}s", flush=True)
                time.sleep(delay)

        if last_error is not None:
            missing.append({"date": current.isoformat(), "source_id": GDELT_SOURCE_ID, "status": "missing", "error_type": type(last_error).__name__, "error": str(last_error)})
            print(f"GDELT gap recorded for {current.isoformat()} after {GDELT_DAY_RETRIES} attempts; continuing chunk", flush=True)

        completed += 1
        if completed == 1 or completed % 25 == 0 or completed == total_days:
            print(f"GDELT staging progress: {completed}/{total_days} days ({completed / total_days:.1%}); restored={restored}", flush=True)
        current += timedelta(days=1)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result = deduplicate_events(combined) if not combined.empty else pd.DataFrame(columns=["event_id", "event_time", "published_at", "available_at", "source_id", "category", "severity", "country", "entity", "sector", "duration_days", "media_intensity", "surprise"])
    if dropped_unavailable:
        print(f"GDELT PIT filter: excluded {dropped_unavailable} rows without a valid event/availability/severity field", flush=True)
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
    print(f"GDELT chunk saved: {args.start} -> {args.end}; rows={len(frame)}; missing_days={len(missing)}; restored/checkpointed={len(list(checkpoint_dir.glob('day_*.csv')))}; path={output}; missing_manifest={missing_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
