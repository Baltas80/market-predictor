"""Source adapters and normalization for the historical staging pipeline.

Network retrieval remains delegated to ``data_sources``. This module converts
source-specific frames into the definitive research schemas without changing
information-availability timestamps.
"""
from __future__ import annotations

import pandas as pd
import requests

from .data_sources import (
    _get,
    load_fred_observations,
    load_gdelt_range,
    load_sec_litigation_releases_rss,
    load_stooq_daily,
    gdelt_events_to_market_events,
)
from .research_schema import deduplicate_events, normalize_event_sources

# DFF is a valid FRED series but does not expose the ALFRED real-time history
# required by this point-in-time pipeline. FEDFUNDS is retained as the
# historical federal-funds-rate proxy with actual vintage support.
FRED_SERIES = ("FEDFUNDS", "DGS10", "CPIAUCSL", "UNRATE", "VIXCLS")
FRED_VINTAGE_DATES_URL = "https://api.stlouisfed.org/fred/series/vintagedates"
FRED_VINTAGE_DATES_PAGE_SIZE = 10000
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
        window_end = min(current + pd.Timedelta(FRED_VINTAGE_CHUNK_DAYS - 1, unit="D"), last)
        windows.append((current.date().isoformat(), window_end.date().isoformat()))
        current = window_end + pd.Timedelta(1, unit="D")
    return windows


def _fred_vintage_dates(api_key: str, series_id: str) -> pd.DatetimeIndex:
    """Return all FRED/ALFRED vintage dates for a series, with pagination.

    The historical PIT anchor may predate the requested dataset start. FRED's
    documentation defines vintage dates as dates when values were revised or
    released, so we retrieve the complete list and let ``fetch_fred`` select
    the vintage that was already in effect at the requested start date.
    """
    offset = 0
    dates: list[str] = []
    while True:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "limit": FRED_VINTAGE_DATES_PAGE_SIZE,
            "offset": offset,
            "sort_order": "asc",
        }
        response = _get(FRED_VINTAGE_DATES_URL, timeout=60, params=params)
        payload = response.json()
        batch = payload.get("vintage_dates", [])
        dates.extend(batch)
        total = int(payload.get("count", len(dates)))
        if not batch or offset + len(batch) >= total:
            break
        offset += len(batch)

    if not dates:
        return pd.DatetimeIndex([], tz="UTC")
    parsed = pd.to_datetime(dates, utc=True, errors="coerce")
    parsed = pd.DatetimeIndex(parsed).dropna().sort_values().unique()
    return pd.DatetimeIndex(parsed)


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
    """Retrieve all required FRED vintages using series-specific PIT availability."""
    if not api_key or not api_key.strip():
        raise ValueError("FRED_API_KEY is missing or empty")

    requested_start = pd.Timestamp(start).normalize()
    requested_end = pd.Timestamp(end).normalize()
    if requested_end < requested_start:
        raise ValueError("FRED end must be on or after start")

    frames: list[pd.DataFrame] = []
    coverage: dict[str, dict[str, str]] = {}
    for series_id in FRED_SERIES:
        try:
            vintage_dates = _fred_vintage_dates(api_key, series_id)
        except requests.HTTPError as exc:
            _raise_fred_error(
                exc,
                series_id=series_id,
                realtime_start=start,
                realtime_end=end,
            )

        if vintage_dates.empty:
            raise RuntimeError(
                f"FRED series {series_id} has no PIT vintage dates; refusing to substitute current revised values."
            )

        prior_or_equal = vintage_dates[vintage_dates <= requested_start.tz_localize("UTC")]
        future = vintage_dates[vintage_dates > requested_start.tz_localize("UTC")]
        if prior_or_equal.size:
            effective_start = requested_start
        elif future.size and future.min().normalize() <= requested_end.tz_localize("UTC"):
            effective_start = future.min().tz_localize(None).normalize()
        else:
            raise RuntimeError(
                f"FRED series {series_id} has no PIT vintage available on or before "
                f"{end}; refusing to backfill the requested range with revised current data."
            )

        coverage[series_id] = {
            "requested_start": requested_start.date().isoformat(),
            "pit_start": effective_start.date().isoformat(),
            "first_vintage": vintage_dates.min().date().isoformat(),
            "last_vintage_in_history": vintage_dates.max().date().isoformat(),
        }

        for realtime_start, realtime_end in _fred_vintage_windows(
            effective_start.date().isoformat(), requested_end.date().isoformat()
        ):
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
    result = result.drop_duplicates(["series_id", "observation_date", "vintage_start"]).sort_values(
        ["series_id", "observation_date", "vintage_start"]
    ).reset_index(drop=True)
    result.attrs["fred_pit_coverage"] = coverage
    return result


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
