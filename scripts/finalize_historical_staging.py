"""Assemble base historical staging plus independently persisted GDELT 1.0 chunks."""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from market_predictor.gdelt1 import GDELT_SOURCE_ID, gdelt1_daily_availability
from market_predictor.historical_coverage import DATASET_START, DATASET_END, GDELT_SOURCE
from market_predictor.historical_ingestion import build_source_manifest, write_source_manifest
from market_predictor.historical_gate import validate_historical_dataset
from market_predictor.research_schema import deduplicate_events

STAGING_VERSION = "2026-09-18-staging-gdelt1-daily-pit-v2"
EVENT_COLUMNS = [
    "event_id", "event_time", "published_at", "available_at", "source_id",
    "category", "severity", "country", "entity", "sector", "duration_days",
    "media_intensity", "surprise",
]
MISSING_COLUMNS = ["date", "source_id", "status", "error_type", "error"]


def _expected_gdelt_chunks() -> list[tuple[int, date, date]]:
    rows: list[tuple[int, date, date]] = []
    current = GDELT_SOURCE.start
    chunk = 1
    while current <= GDELT_SOURCE.end:
        chunk_end = min(current + timedelta(days=27), GDELT_SOURCE.end)
        rows.append((chunk, current, chunk_end))
        current = chunk_end + timedelta(days=1)
        chunk += 1
    return rows


def read_missing_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"GDELT completeness gate: required gap manifest is missing: {path}")
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

    expected_chunks = _expected_gdelt_chunks()
    data_chunk_paths: list[Path] = []
    missing_paths: list[Path] = []
    for chunk, start, end in expected_chunks:
        data_path = chunks_dir / f"gdelt_{chunk}_{start.isoformat()}_{end.isoformat()}.csv"
        missing_path = chunks_dir / f"gdelt_{chunk}_missing.csv"
        if not data_path.exists():
            raise RuntimeError(f"GDELT completeness gate: missing expected data chunk: {data_path.name}")
        if not missing_path.exists():
            raise RuntimeError(f"GDELT completeness gate: missing expected gap manifest: {missing_path.name}")
        data_chunk_paths.append(data_path)
        missing_paths.append(missing_path)

    extra_data = sorted(
        p for p in chunks_dir.glob("gdelt_*.csv")
        if not p.stem.endswith("_missing") and p not in data_chunk_paths and p.name != "gdelt_recovered.csv"
    )
    if extra_data:
        raise RuntimeError(
            "GDELT completeness gate: unexpected data chunk artifacts present: "
            + ", ".join(p.name for p in extra_data)
        )

    frames = [pd.read_csv(path) for path in data_chunk_paths]
    recovered_path = chunks_dir / "gdelt_recovered.csv"
    if recovered_path.exists() and recovered_path.stat().st_size > 0:
        frames.append(pd.read_csv(recovered_path))

    nonempty_frames = [frame for frame in frames if not frame.empty]
    gdelt = (
        deduplicate_events(pd.concat(nonempty_frames, ignore_index=True))
        if nonempty_frames
        else pd.DataFrame(columns=EVENT_COLUMNS)
    )

    if gdelt.empty:
        raise RuntimeError("GDELT completeness gate: no GDELT 1.0 event rows were recovered")

    if "source_id" not in gdelt.columns:
        raise RuntimeError("GDELT identity gate: source_id column is absent")
    bad_source_ids = sorted(set(gdelt["source_id"].dropna().astype(str)) - {GDELT_SOURCE_ID})
    if bad_source_ids:
        raise RuntimeError(
            "GDELT identity gate: non-canonical source identifiers found: "
            + ", ".join(bad_source_ids)
        )

    gdelt.to_csv(raw / "events_gdelt.csv", index=False, lineterminator="\n")
    gdelt.to_csv(normalized / "events_gdelt.csv", index=False, lineterminator="\n")

    missing_frames = [read_missing_manifest(path) for path in missing_paths]
    recovered_missing_path = chunks_dir / "gdelt_recovery_missing.csv"
    if recovered_missing_path.exists():
        missing_frames.append(read_missing_manifest(recovered_missing_path))

    nonempty_missing = [frame for frame in missing_frames if not frame.empty]
    missing = (
        pd.concat(nonempty_missing, ignore_index=True)
        if nonempty_missing
        else pd.DataFrame(columns=MISSING_COLUMNS)
    )
    if not missing.empty:
        missing = missing.drop_duplicates(subset=["date", "source_id"], keep="last").sort_values(
            ["date", "source_id"]
        )
        bad_missing_source_ids = sorted(
            set(missing["source_id"].dropna().astype(str)) - {GDELT_SOURCE_ID}
        )
        if bad_missing_source_ids:
            raise RuntimeError(
                "GDELT identity gate: gap manifests contain non-canonical source identifiers: "
                + ", ".join(bad_missing_source_ids)
            )

    missing_path = raw / "events_gdelt_missing.csv"
    missing.to_csv(missing_path, index=False, lineterminator="\n")

    gdelt_missing = missing.loc[missing["source_id"] == GDELT_SOURCE_ID].copy()
    if not gdelt_missing.empty:
        dates = sorted(
            pd.to_datetime(gdelt_missing["date"], errors="coerce").dropna().dt.date.unique()
        )
        preview = ", ".join(day.isoformat() for day in dates[:10])
        suffix = " ..." if len(dates) > 10 else ""
        raise RuntimeError(
            f"Historical staging is not admissible: {len(gdelt_missing)} GDELT 1.0 source-days remain missing "
            f"within required coverage {GDELT_SOURCE.start.isoformat()} -> {GDELT_SOURCE.end.isoformat()}; "
            f"examples: {preview}{suffix}. No final lockbox or financial report will run."
        )

    manifests = []
    for source_id, source_type, path, uri, policy in (
        (
            "Stooq_SPX",
            "market",
            normalized / "market.csv",
            "https://stooq.com/q/d/l/",
            "daily cash-session close represented in UTC",
        ),
        (
            "FRED_required_series",
            "macro",
            raw / "macro_fred.csv",
            "https://api.stlouisfed.org/fred/series/observations",
            "FRED realtime_start/vintage_start discovered from series/vintagedates; conservative decision-time lag is applied downstream",
        ),
        (
            GDELT_SOURCE_ID,
            "events",
            normalized / "events_gdelt.csv",
            "https://data.gdeltproject.org/events/{date}.export.CSV.zip",
            "GDELT 1.0 daily file publication boundary: conservative next-day 06:00 America/New_York publication boundary; original article publication time is unknown",
        ),
    ):
        frame = pd.read_csv(path)
        if source_id == "Stooq_SPX":
            frame = market
        manifests.append(
            build_source_manifest(
                frame,
                source_id=source_id,
                source_type=source_type,
                retrieval_version=STAGING_VERSION,
                source_uri=uri,
                availability_policy=policy,
            )
        )

    sec_path = normalized / "events_sec_litigation.csv"
    if sec_path.exists() and sec_path.stat().st_size > 0:
        sec = pd.read_csv(sec_path)
        if not sec.empty:
            manifests.append(
                build_source_manifest(
                    sec,
                    source_id="SEC_Litigation_Releases",
                    source_type="events",
                    retrieval_version=STAGING_VERSION,
                    source_uri="https://www.sec.gov/enforcement-litigation/litigation-releases/rss",
                    availability_policy="RSS publication time used as availability time",
                )
            )

    events = gdelt
    if sec_path.exists() and sec_path.stat().st_size > 0:
        sec = pd.read_csv(sec_path)
        if not sec.empty:
            events = pd.concat([events, sec], ignore_index=True)

    gate = validate_historical_dataset(
        market,
        macro=macro,
        events=events,
        manifests=manifests,
    )
    limitations = [
        "GDELT 1.0 daily event coverage begins in 2015-02-19; it is not a 2000-2015-02-18 event source",
        "GDELT 1.0 publication time is not stored per event; availability uses the conservative next-day 06:00 America/New_York boundary",
        "SEC adapter is an RSS snapshot and does not provide a verified 2000-2025 archive",
    ]
    result = {
        "status": "admissible_with_source_limits",
        "staging_version": STAGING_VERSION,
        "dataset_start": DATASET_START.isoformat(),
        "dataset_end": DATASET_END.isoformat(),
        "historical_gate": gate,
        "gdelt_chunk_count": len(data_chunk_paths),
        "gdelt_rows": len(gdelt),
        "gdelt_missing_day_count": 0,
        "gdelt_missing_manifest": str(missing_path),
        "gdelt_expected_chunk_count": len(expected_chunks),
        "gdelt_expected_chunk_coverage": [
            {"chunk": chunk, "start": start.isoformat(), "end": end.isoformat()}
            for chunk, start, end in expected_chunks
        ],
        "limitations": limitations,
        "manifest_count": len(manifests),
        "manifest": str(staging / "source_manifest.json"),
        "normalized_directory": str(normalized),
    }
    write_source_manifest(manifests, str(staging / "source_manifest.json"))
    (staging / "staging_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
