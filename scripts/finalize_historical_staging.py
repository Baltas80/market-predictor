"""Assemble base historical staging plus independently persisted GDELT chunks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from market_predictor.historical_coverage import DATASET_START, DATASET_END
from market_predictor.historical_ingestion import build_source_manifest, write_source_manifest
from market_predictor.historical_gate import validate_historical_dataset
from market_predictor.research_schema import deduplicate_events

STAGING_VERSION = "2026-09-09-staging-weekly-gdelt-with-gap-manifest-v4"
EVENT_COLUMNS = [
    "event_id", "event_time", "published_at", "available_at", "source_id",
    "category", "severity", "country", "entity", "sector", "duration_days",
    "media_intensity", "surprise",
]
MISSING_COLUMNS = ["date", "source_id", "status", "error_type", "error"]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_missing_manifest(path: Path) -> pd.DataFrame:
    """Read a chunk gap manifest, treating an empty file as no gaps."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=MISSING_COLUMNS)
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=MISSING_COLUMNS)
    if frame.empty:
        return pd.DataFrame(columns=MISSING_COLUMNS)
    missing = set(MISSING_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"GDELT gap manifest missing columns: {sorted(missing)}")
    return frame[MISSING_COLUMNS].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", default="data/historical")
    parser.add_argument("--chunks", default="data/gdelt_chunks")
    args = parser.parse_args()

    staging = Path(args.staging)
    chunks_dir = Path(args.chunks)
    normalized = staging / "normalized"
    raw = staging / "raw"
    normalized.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    market = pd.read_csv(normalized / "market.csv", index_col=0, parse_dates=[0])
    market.index.name = "date"
    macro = pd.read_csv(raw / "macro_fred.csv")

    chunk_paths = sorted(chunks_dir.glob("gdelt_*.csv"))
    data_chunk_paths = [p for p in chunk_paths if not p.stem.endswith("_missing")]
    if not data_chunk_paths:
        raise RuntimeError("No persisted GDELT data chunks found")
    frames = [pd.read_csv(path) for path in data_chunk_paths]
    nonempty_frames = [frame for frame in frames if not frame.empty]
    if nonempty_frames:
        gdelt = deduplicate_events(pd.concat(nonempty_frames, ignore_index=True))
    else:
        gdelt = pd.DataFrame(columns=EVENT_COLUMNS)
    gdelt.to_csv(raw / "events_gdelt.csv", index=False, lineterminator="\n")
    gdelt.to_csv(normalized / "events_gdelt.csv", index=False, lineterminator="\n")

    missing_paths = sorted(chunks_dir.glob("gdelt_*_missing.csv"))
    missing_frames = [read_missing_manifest(path) for path in missing_paths]
    nonempty_missing = [frame for frame in missing_frames if not frame.empty]
    missing = pd.concat(nonempty_missing, ignore_index=True) if nonempty_missing else pd.DataFrame(columns=MISSING_COLUMNS)
    if not missing.empty:
        missing = missing.drop_duplicates(subset=["date", "source_id"], keep="last").sort_values(["date", "source_id"])
    missing_path = raw / "events_gdelt_missing.csv"
    missing.to_csv(missing_path, index=False, lineterminator="\n")

    manifests = []
    for source_id, source_type, path, uri, policy in (
        ("Stooq_SPX", "market", normalized / "market.csv", "https://stooq.com/q/d/l/", "daily cash-session close represented in UTC"),
        ("FRED_required_series", "macro", raw / "macro_fred.csv", "https://api.stlouisfed.org/fred/series/observations", "FRED realtime_start/vintage_start discovered from series/vintagedates; conservative decision-time lag is applied downstream"),
        ("GDELT_2_Event_Database", "events", normalized / "events_gdelt.csv", "https://data.gdeltproject.org/events/{date}.export.CSV.zip", "DATEADDED is retained as availability proxy; publication time is unknown"),
    ):
        frame = pd.read_csv(path)
        if source_id == "Stooq_SPX":
            frame = market
        manifests.append(build_source_manifest(frame, source_id=source_id, source_type=source_type, retrieval_version=STAGING_VERSION, source_uri=uri, availability_policy=policy))

    sec_path = normalized / "events_sec_litigation.csv"
    if sec_path.exists() and sec_path.stat().st_size > 0:
        sec = pd.read_csv(sec_path)
        if not sec.empty:
            manifests.append(build_source_manifest(sec, source_id="SEC_Litigation_Releases", source_type="events", retrieval_version=STAGING_VERSION, source_uri="https://www.sec.gov/enforcement-litigation/litigation-releases/rss", availability_policy="RSS publication time used as availability time"))

    events = gdelt
    if sec_path.exists() and sec_path.stat().st_size > 0:
        sec = pd.read_csv(sec_path)
        if not sec.empty:
            events = pd.concat([events, sec], ignore_index=True)

    gate = validate_historical_dataset(market, macro=macro, events=events, manifests=manifests)
    limitations = [
        "GDELT historical event coverage begins in 2015; it is not a 2000-2014 event source",
        "SEC adapter is an RSS snapshot and does not provide a verified 2000-2025 archive",
    ]
    if not missing.empty:
        limitations.append(f"GDELT has {len(missing)} source-days currently missing; they are recorded for later recovery and are never imputed")
    result = {
        "status": "admissible_with_source_limits",
        "staging_version": STAGING_VERSION,
        "dataset_start": DATASET_START.isoformat(),
        "dataset_end": DATASET_END.isoformat(),
        "historical_gate": gate,
        "gdelt_chunk_count": len(data_chunk_paths),
        "gdelt_rows": len(gdelt),
        "gdelt_missing_day_count": len(missing),
        "gdelt_missing_manifest": str(missing_path),
        "limitations": limitations,
        "manifest_count": len(manifests),
        "manifest": str(staging / "source_manifest.json"),
        "normalized_directory": str(normalized),
    }
    write_source_manifest(manifests, str(staging / "source_manifest.json"))
    (staging / "staging_result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
