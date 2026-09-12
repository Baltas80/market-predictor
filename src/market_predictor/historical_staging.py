"""Reproducible staging orchestration for the 2000-2025 research dataset."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from .historical_adapters import fetch_fred, fetch_gdelt, fetch_market, fetch_sec
from .historical_ingestion import SourceManifest, build_source_manifest, write_source_manifest
from .historical_coverage import DATASET_START, DATASET_END, GDELT_SOURCE
from .historical_gate import validate_historical_dataset


STAGING_VERSION = "2026-09-06-staging-v2"


def _write_frame(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z")


def _read_market(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=[0])
    frame.index.name = "date"
    return frame


def _coverage_manifest(frame: pd.DataFrame, *, source_id: str, source_type: str, source_uri: str, availability_policy: str) -> SourceManifest:
    return build_source_manifest(
        frame,
        source_id=source_id,
        source_type=source_type,
        retrieval_version=STAGING_VERSION,
        source_uri=source_uri,
        availability_policy=availability_policy,
    )


def stage_historical(
    output_dir: Path,
    *,
    fred_api_key: str | None,
    include_gdelt: bool = True,
    include_sec: bool = True,
    market_fetcher: Callable[[str, str], pd.DataFrame] = fetch_market,
    fred_fetcher: Callable[[str, str, str], pd.DataFrame] = fetch_fred,
    gdelt_fetcher: Callable[[str, str], pd.DataFrame] = fetch_gdelt,
    sec_fetcher: Callable[[], pd.DataFrame] = fetch_sec,
) -> dict[str, object]:
    """Execute all staging phases and admit data only after Historical Gate.

    The function is dependency-injected for deterministic CI tests. Production
    callers use the real Stooq/Yahoo, FRED, GDELT and SEC adapters by default.
    """
    if not fred_api_key:
        raise RuntimeError("FRED_API_KEY is required for historical staging")

    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    normalized_dir = output_dir / "normalized"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    stages = [{"name": name, "status": "blocked"} for name in (
        "download", "raw", "hash", "normalize", "pit", "historical_gate", "dataset"
    )]
    stage_map = {stage["name"]: stage for stage in stages}
    manifests: list[SourceManifest] = []
    limitations: list[str] = []

    market = market_fetcher(DATASET_START.isoformat(), DATASET_END.isoformat())
    if market.empty:
        raise RuntimeError("Market adapter returned no market observations")
    market_source_id = str(market.attrs.get("source_id", "Stooq_SPX"))
    if market_source_id == "YahooFinance_GSPC":
        market_source_uri = "https://query1.finance.yahoo.com/v8/finance/chart/^GSPC"
        limitations.append(
            "Stooq market retrieval failed or returned invalid OHLCV; S&P 500 OHLCV was retrieved from Yahoo Finance chart API fallback"
        )
    else:
        market_source_uri = "https://stooq.com/q/d/l/"
    _write_frame(market, raw_dir / "market.csv", index=True)
    stage_map["download"]["status"] = "complete"
    stage_map["raw"]["status"] = "complete"
    market_manifest = _coverage_manifest(
        market,
        source_id=market_source_id,
        source_type="market",
        source_uri=market_source_uri,
        availability_policy="daily cash-session close represented in UTC",
    )
    manifests.append(market_manifest)
    _write_frame(market, normalized_dir / "market.csv", index=True)

    macro = fred_fetcher(fred_api_key, DATASET_START.isoformat(), DATASET_END.isoformat())
    if macro.empty:
        raise RuntimeError("FRED returned no required macro vintages")
    fred_pit_coverage = macro.attrs.get("fred_pit_coverage", {})
    for series_id, details in fred_pit_coverage.items():
        pit_start = details.get("pit_start") if isinstance(details, dict) else None
        if pit_start and pit_start > DATASET_START.isoformat():
            limitations.append(
                f"FRED {series_id} PIT history starts on {pit_start}; observations before that date are unavailable from FRED/ALFRED and were not backfilled with revised values"
            )
    _write_frame(macro, raw_dir / "macro_fred.csv")
    macro_manifest = _coverage_manifest(
        macro,
        source_id="FRED_required_series",
        source_type="macro",
        source_uri="https://api.stlouisfed.org/fred/series/observations",
        availability_policy="FRED realtime_start/vintage_start discovered from series/vintagedates; conservative decision-time lag is applied downstream",
    )
    manifests.append(macro_manifest)
    _write_frame(macro, normalized_dir / "macro.csv")
    stage_map["normalize"]["status"] = "complete"
    stage_map["pit"]["status"] = "complete"

    event_frames: list[pd.DataFrame] = []
    if include_gdelt:
        gdelt = gdelt_fetcher(GDELT_SOURCE.start.isoformat(), DATASET_END.isoformat())
        _write_frame(gdelt, raw_dir / "events_gdelt.csv")
        _write_frame(gdelt, normalized_dir / "events_gdelt.csv")
        if not gdelt.empty:
            manifests.append(_coverage_manifest(
                gdelt,
                source_id="GDELT_2_Event_Database",
                source_type="events",
                source_uri="https://data.gdeltproject.org/events/{date}.export.CSV.zip",
                availability_policy="DATEADDED is retained as availability proxy; publication time is unknown",
            ))
            event_frames.append(gdelt)
        else:
            limitations.append("GDELT produced no events")
    if include_sec:
        sec = sec_fetcher()
        _write_frame(sec, raw_dir / "events_sec_litigation.csv")
        _write_frame(sec, normalized_dir / "events_sec_litigation.csv")
        if not sec.empty:
            manifests.append(_coverage_manifest(
                sec,
                source_id="SEC_Litigation_Releases",
                source_type="events",
                source_uri="https://www.sec.gov/enforcement-litigation/litigation-releases/rss",
                availability_policy="RSS publication time used as availability time",
            ))
            event_frames.append(sec)
        limitations.append("SEC adapter is an RSS snapshot and does not provide a verified 2000-2025 archive")
    if include_gdelt:
        limitations.append("GDELT historical event coverage begins in 2015-02-19; it is not a 2000-2015-02-18 event source")

    stage_map["hash"]["status"] = "complete"
    events = pd.concat(event_frames, ignore_index=True) if event_frames else None
    gate = validate_historical_dataset(market, macro=macro, events=events, manifests=manifests)
    stage_map["historical_gate"]["status"] = "complete"

    write_source_manifest(manifests, str(output_dir / "source_manifest.json"))
    dataset_status = "admissible_with_source_limits" if limitations else "admissible"
    stage_map["dataset"]["status"] = dataset_status

    result = {
        "status": dataset_status,
        "staging_version": STAGING_VERSION,
        "dataset_start": DATASET_START.isoformat(),
        "dataset_end": DATASET_END.isoformat(),
        "stages": stages,
        "historical_gate": gate,
        "limitations": limitations,
        "fred_pit_coverage": fred_pit_coverage,
        "manifest_count": len(manifests),
        "manifest": str(output_dir / "source_manifest.json"),
        "normalized_directory": str(normalized_dir),
    }
    (output_dir / "staging_result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return result
