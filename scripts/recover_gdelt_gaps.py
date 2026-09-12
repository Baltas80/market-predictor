"""Recover only source-days recorded as missing by the GDELT ingestion pass.

Recovery is deliberately gap-driven: it never reruns successful days and never
fills a missing day with an estimate. A day that remains unavailable is kept in
a deterministic residual gap manifest for the historical gate.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import zipfile

import pandas as pd
import requests

try:
    from scripts.ingest_gdelt_chunk import load_gdelt_day
except ModuleNotFoundError:
    from ingest_gdelt_chunk import load_gdelt_day
from market_predictor.research_schema import deduplicate_events, normalize_event_sources

RECOVERY_RETRIES = 4
RECOVERY_RETRY_BASE_SECONDS = 2
MISSING_COLUMNS = ["date", "source_id", "status", "error_type", "error"]
EVENT_COLUMNS = [
    "event_id", "event_time", "published_at", "available_at", "source_id",
    "category", "severity", "country", "entity", "sector", "duration_days",
    "media_intensity", "surprise",
]


def read_missing(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=MISSING_COLUMNS)
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=MISSING_COLUMNS)
    missing = set(MISSING_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing-manifest missing columns: {sorted(missing)}")
    return frame[MISSING_COLUMNS].copy()


def recover_day(day: str) -> tuple[pd.DataFrame | None, dict[str, str] | None]:
    last_error: Exception | None = None
    for attempt in range(1, RECOVERY_RETRIES + 1):
        try:
            raw = load_gdelt_day(day)
            if raw.empty:
                raise ValueError(f"GDELT source-day {day} returned an empty export")
            if "event_id" not in raw and "global_event_id" in raw:
                raw = raw.rename(columns={"global_event_id": "event_id"})
            if "event_time" not in raw and "sql_date" in raw:
                raw["event_time"] = raw["sql_date"]
            # GDELT DATEADDED is an information-availability timestamp, not
            # article publication time. Publication therefore remains unknown.
            if "published_at" not in raw:
                raw["published_at"] = pd.NaT
            if "available_at" not in raw and "date_added" in raw:
                raw["available_at"] = raw["date_added"]
            normalized = normalize_event_sources(raw, source_id="GDELT_2_Event_Database")
            before = len(normalized)
            normalized = normalized.dropna(
                subset=["event_id", "event_time", "available_at", "severity"]
            ).copy()
            if before and normalized.empty:
                raise ValueError(
                    f"GDELT PIT recovery rejected every row for {day}; "
                    "required event/availability/severity fields are invalid"
                )
            return normalized, None
        except (requests.RequestException, TimeoutError, ValueError, zipfile.BadZipFile) as exc:
            last_error = exc
            if attempt < RECOVERY_RETRIES:
                delay = RECOVERY_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                print(f"Recovery retry {attempt}/{RECOVERY_RETRIES - 1} for {day} after {type(exc).__name__}; waiting {delay}s", flush=True)
                time.sleep(delay)
    assert last_error is not None
    return None, {
        "date": pd.Timestamp(day).date().isoformat(),
        "source_id": "GDELT_2_Event_Database",
        "status": "missing",
        "error_type": type(last_error).__name__,
        "error": str(last_error),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/gdelt_chunks")
    parser.add_argument("--output", default="data/gdelt_chunks/gdelt_recovered.csv")
    parser.add_argument("--remaining-output", default="data/gdelt_chunks/gdelt_recovery_missing.csv")
    args = parser.parse_args()

    chunks_dir = Path(args.chunks)
    manifests = sorted(chunks_dir.glob("gdelt_*_missing.csv"))
    frames: list[pd.DataFrame] = []
    remaining: list[dict[str, str]] = []
    gaps: set[tuple[str, str]] = set()
    for manifest in manifests:
        frame = read_missing(manifest)
        for row in frame.to_dict("records"):
            gaps.add((str(row["date"]), str(row["source_id"])))

    print(f"GDELT recovery: {len(gaps)} unique source-days queued", flush=True)
    for date, source_id in sorted(gaps):
        if source_id != "GDELT_2_Event_Database":
            remaining.append({
                "date": date,
                "source_id": source_id,
                "status": "missing",
                "error_type": "UnsupportedRecoverySource",
                "error": "Recovery supports GDELT source-days only",
            })
            continue
        frame, residual = recover_day(date)
        if frame is not None:
            frames.append(frame)
            print(f"GDELT recovered: {date}", flush=True)
        else:
            assert residual is not None
            remaining.append(residual)
            print(f"GDELT remains missing: {date}", flush=True)

    recovered = deduplicate_events(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame(columns=EVENT_COLUMNS)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    recovered.to_csv(output, index=False, lineterminator="\n")

    residual = pd.DataFrame(remaining, columns=MISSING_COLUMNS)
    if not residual.empty:
        residual = residual.drop_duplicates(subset=["date", "source_id"], keep="last").sort_values(["date", "source_id"])
    remaining_output = Path(args.remaining_output)
    remaining_output.parent.mkdir(parents=True, exist_ok=True)
    residual.to_csv(remaining_output, index=False, lineterminator="\n")

    print(f"GDELT recovery complete: recovered_days={len(gaps) - len(residual)}; remaining_days={len(residual)}; recovered_rows={len(recovered)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
