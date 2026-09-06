"""Source adapters and normalization for the historical staging pipeline.

Network retrieval remains delegated to ``data_sources``. This module converts
source-specific frames into the definitive research schemas without changing
information-availability timestamps.
"""
from __future__ import annotations

import pandas as pd
import requests

from .data_sources import (
    load_fred_observations,
    load_gdelt_range,
    load_sec_litigation_releases_rss,
    load_stooq_daily,
    gdelt_events_to_market_events,
)
from .research_schema import deduplicate_events, normalize_event_sources

FRED_SERIES = ("DFF", "FEDFUNDS", "DGS10", "CPIAUCSL", "UNRATE", "VIXCLS")
FRED_VINTAGE_CHUNK_DAYS = 365


def fetch_market(start: str, end: str) -> pd.DataFrame:
    """Retrieve the canonical Stooq daily market frame."""
    return load_stooq_daily("^spx", start=start, end=end)


def _fred_vintage_windows(start: str, end: str) -> list[tuple[str, str]]:
    """Split a long real-time period into bounded FRED vintage windows."""
    first = pd.Timestamp(start).normalize()
    last = pd.Timestamp(end).normalize()
    if last < first:
        raise ValueError("FRED end must be on or after start")
    windows: list[tuple[str, str]] = []
    current = first
    while current <= last:
        window_end = min(current + pd.Timedelta(days=FRED_VINTAGE_CHUNK_DAYS - 1), last)
        windows.append((current.date().isoformat(), window_end.date().isoformat()))
        current = window_end + pd.Timedelta(days=1)
    return windows


def _raise_fred_error(exc: requests.HTTPError, *, series_id: str, realtime_start: str, realtime_end: str) -> None:
    """Replace opaque HTTP 400 errors with FRED's actual API error message."""
    response = exc.response
    status = response.status_code if response is not None else "unknown"
    detail = ""
    if response is not None:
        try:
            payload = response.json()
            detail = str(payload.get("error_message") or payload.get("message") or payload.get("error") or "")
        except ValueError:
            detail = response.text.strip()
    if not detail:
        detail = str(exc)
    raise RuntimeError(
        f"FRED request failed: series_id={series_id}, realtime_start={realtime_start}, "
        f"realtime_end={realtime_end}, status={status}, detail={detail[:1000]}"
    ) from exc


def fetch_fred(api_key: str, start: str, end: str) -> pd.DataFrame:
    """Retrieve all required FRED vintages and normalize to the PIT schema."""
    if not api_key or not api_key.strip():
        raise ValueError("FRED_API_KEY is missing or empty")

    frames: list[pd.DataFrame] = []
    for series_id in FRED_SERIES:
        for realtime_start, realtime_end in _fred_vintage_windows(start, end):
            try:
                frame = load_fred_observations(
                    series_id,
                    api_key,
                    realtime_start=realtime_start,
                    realtime_end=realtime_end,
                ).copy()
            except requests.HTTPError as exc:
                _raise_fred_error(
                    exc,
                    series_id=series_id,
                    realtime_start=realtime_start,
                    realtime_end=realtime_end,
                )
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
