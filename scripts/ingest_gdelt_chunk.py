"""Download and persist one bounded GDELT historical chunk.

Each chunk is independently reproducible so a failed/cancelled CI job never
requires re-downloading earlier completed chunks. Events without a usable
GDELT DATEADDED timestamp are excluded because their information-availability
cutoff cannot be established safely for point-in-time research.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import time
import zipfile

import pandas as pd
import requests

from market_predictor.data_sources import gdelt_events_to_market_events, load_gdelt_day
from market_predictor.research_schema import deduplicate_events, normalize_event_sources

GDELT_DAY_RETRIES = 4
GDELT_RETRY_BASE_SECONDS = 2


def fetch_gdelt_chunk(start: str, end: str) -> pd.DataFrame:
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    if end_date < start_date:
        raise ValueError("GDELT end must be on or after start")

    frames: list[pd.DataFrame] = []
    total_days = (end_date - start_date).days + 1
    current = start_date
    completed = 0
    dropped_unavailable = 0

    while current <= end_date:
        last_error: Exception | None = None
        for attempt in range(1, GDELT_DAY_RETRIES + 1):
            try:
                raw_day = load_gdelt_day(current)
                normalized = gdelt_events_to_market_events(raw_day)
                normalized = normalize_event_sources(
                    normalized, source_id="GDELT_2_Event_Database"
                )
                before = len(normalized)
                normalized = normalized.dropna(
                    subset=["event_id", "event_time", "available_at", "severity"]
                ).copy()
                dropped_unavailable += before - len(normalized)
                frames.append(normalized)
                last_error = None
                break
            except (requests.RequestException, TimeoutError, ValueError, zipfile.BadZipFile) as exc:
                last_error = exc
                if attempt >= GDELT_DAY_RETRIES:
                    break
                delay = GDELT_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                print(
                    f"GDELT retry {attempt}/{GDELT_DAY_RETRIES - 1} for {current.isoformat()} "
                    f"after {type(exc).__name__}; waiting {delay}s",
                    flush=True,
                )
                time.sleep(delay)

        if last_error is not None:
            raise RuntimeError(
                f"GDELT download failed permanently for {current.isoformat()} "
                f"after {GDELT_DAY_RETRIES} attempts: {last_error}"
            ) from last_error

        completed += 1
        if completed == 1 or completed % 25 == 0 or completed == total_days:
            print(
                f"GDELT staging progress: {completed}/{total_days} days "
                f"({completed / total_days:.1%})",
                flush=True,
            )
        current += timedelta(days=1)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result = deduplicate_events(combined)
    if dropped_unavailable:
        print(
            f"GDELT PIT filter: excluded {dropped_unavailable} rows without a valid "
            "event/availability/severity field",
            flush=True,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = fetch_gdelt_chunk(args.start, args.end)
    frame.to_csv(output, index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z")
    print(
        f"GDELT chunk saved: {args.start} -> {args.end}; rows={len(frame)}; path={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
