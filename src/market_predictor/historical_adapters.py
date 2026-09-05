"""Source adapters and normalization for the historical staging pipeline.

Network retrieval remains delegated to ``data_sources``. This module converts
source-specific frames into the definitive research schemas without changing
information-availability timestamps.
"""
from __future__ import annotations

import pandas as pd

from .data_sources import (
    load_fred_observations,
    load_gdelt_range,
    load_sec_litigation_releases_rss,
    load_stooq_daily,
    gdelt_events_to_market_events,
)
from .research_schema import deduplicate_events, normalize_event_sources

FRED_SERIES = ("DFF", "FEDFUNDS", "DGS10", "CPIAUCSL", "UNRATE", "VIXCLS")


def fetch_market(start: str, end: str) -> pd.DataFrame:
    """Retrieve the canonical Stooq daily market frame."""
    return load_stooq_daily("^spx", start=start, end=end)


def fetch_fred(api_key: str, start: str, end: str) -> pd.DataFrame:
    """Retrieve all required FRED vintages and normalize to the PIT schema."""
    frames: list[pd.DataFrame] = []
    for series_id in FRED_SERIES:
        frame = load_fred_observations(
            series_id,
            api_key,
            realtime_start=start,
            realtime_end=end,
        ).copy()
        if frame.empty:
            continue
        frame["series_id"] = series_id
        frame["observation_date"] = pd.to_datetime(frame["date"], utc=True).dt.normalize()
        frame["vintage_start"] = pd.to_datetime(frame["realtime_start"], utc=True).dt.normalize()
        frame["vintage_end"] = pd.to_datetime(frame["realtime_end"], utc=True).dt.normalize()
        frames.append(frame[["series_id", "observation_date", "value", "vintage_start", "vintage_end"]])
    if not frames:
        return pd.DataFrame(columns=["series_id", "observation_date", "value", "vintage_start", "vintage_end"])
    result = pd.concat(frames, ignore_index=True)
    return result.drop_duplicates(["series_id", "observation_date", "vintage_start"]).sort_values(
        ["series_id", "observation_date", "vintage_start"]
    ).reset_index(drop=True)


def fetch_gdelt(start: str, end: str) -> pd.DataFrame:
    """Retrieve the GDELT event range and preserve DATEADDED as availability proxy."""
    raw = load_gdelt_range(start, end)
    normalized = gdelt_events_to_market_events(raw)
    return deduplicate_events(normalize_event_sources(normalized, source_id="GDELT_2_Event_Database"))


def fetch_sec() -> pd.DataFrame:
    """Retrieve the SEC litigation RSS feed.

    The feed is a current-source snapshot, not a 2000-2025 historical archive;
    callers must record that coverage limitation in staging metadata.
    """
    raw = load_sec_litigation_releases_rss()
    return deduplicate_events(normalize_event_sources(raw, source_id="SEC_Litigation_Releases"))
