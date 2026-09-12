"""Assemble base historical staging with GDELT 2.0 BigQuery events."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from market_predictor.historical_coverage import DATASET_END, DATASET_START
from market_predictor.historical_ingestion import build_source_manifest, write_source_manifest
from market_predictor.historical_gate import validate_historical_dataset
from market_predictor.research_schema import deduplicate_events

STAGING_VERSION = "2026-09-13-staging-gdelt2-bigquery-pit-v1"
GDELT_SOURCE_ID = "GDELT_2_Event_Database_BigQuery"
GDELT_URI = "https://gdelt-bq.gdeltproject.org/"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", default="data/historical")
    parser.add_argument("--events", required=True)
    args = parser.parse_args()

    staging = Path(args.staging)
    normalized = staging / "normalized"
    raw = staging / "raw"
    normalized.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    market = pd.read_csv(normalized / "market.csv", index_col=0, parse_dates=[0])
    market.index.name = "date"
    macro = pd.read_csv(raw / "macro_fred.csv")

    events_path = Path(args.events)
    gdelt = pd.read_csv(events_path, compression="infer")
    if gdelt.empty:
        raise RuntimeError("GDELT2 staging is empty")
    gdelt = deduplicate_events(gdelt)
    if gdelt.empty:
        raise RuntimeError("GDELT2 staging contains no events after deduplication")
    if set(gdelt["source_id"].dropna().unique()) != {GDELT_SOURCE_ID}:
        raise RuntimeError("GDELT2 staging contains unexpected source IDs")

    for target in (raw / "events_gdelt.csv.gz", normalized / "events_gdelt.csv.gz"):
        gdelt.to_csv(target, index=False, compression="gzip", lineterminator="\n")

    missing_path = raw / "events_gdelt_missing.csv"
    pd.DataFrame(columns=["date", "source_id", "status", "error_type", "error"]).to_csv(
        missing_path, index=False, lineterminator="\n"
    )

    manifests = [
        build_source_manifest(
            market,
            source_id="Stooq_SPX",
            source_type="market",
            retrieval_version=STAGING_VERSION,
            source_uri="https://stooq.com/q/d/l/",
            availability_policy="daily cash-session close represented in UTC",
        ),
        build_source_manifest(
            macro,
            source_id="FRED_required_series",
            source_type="macro",
            retrieval_version=STAGING_VERSION,
            source_uri="https://api.stlouisfed.org/fred/series/observations",
            availability_policy="FRED realtime_start/vintage_start discovered from series/vintagedates; conservative decision-time lag is applied downstream",
        ),
        build_source_manifest(
            gdelt,
            source_id=GDELT_SOURCE_ID,
            source_type="events",
            retrieval_version=STAGING_VERSION,
            source_uri=GDELT_URI,
            availability_policy="GDELT 2.0 DATEADDED (UTC) is the information-availability timestamp; SQLDATE is event date only",
        ),
    ]

    events = gdelt
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
            events = pd.concat([events, sec], ignore_index=True)

    gate = validate_historical_dataset(market, macro=macro, events=events, manifests=manifests)
    limitations = [
        "GDELT 2.0 event coverage begins on 2015-02-19; there is no GDELT2 event source before that date",
        "GDELT2 DATEADDED is the master-database availability timestamp, not the original article publication timestamp",
        "The BigQuery partition filter is used only for scan pruning; the PIT window is explicitly enforced on DATEADDED",
        "SEC adapter is an RSS snapshot and does not provide a verified 2000-2025 archive",
    ]
    result = {
        "status": "admissible_with_source_limits",
        "staging_version": STAGING_VERSION,
        "dataset_start": DATASET_START.isoformat(),
        "dataset_end": DATASET_END.isoformat(),
        "historical_gate": gate,
        "gdelt_source": GDELT_SOURCE_ID,
        "gdelt_rows": len(gdelt),
        "gdelt_missing_day_count": 0,
        "gdelt_missing_manifest": str(missing_path),
        "limitations": limitations,
        "manifest_count": len(manifests),
        "manifest": str(staging / "source_manifest.json"),
        "normalized_directory": str(normalized),
    }
    write_source_manifest(manifests, str(staging / "source_manifest.json"))
    (staging / "staging_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
