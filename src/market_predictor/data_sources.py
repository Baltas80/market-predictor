"""Adapters for reproducible real-world market, macro and event data."""

from __future__ import annotations

from datetime import timedelta
from io import BytesIO, StringIO
from pathlib import Path
import hashlib
import time
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import requests

from .fred_pit import audit_fred_point_in_time
from .session_calendar import session_close

STOOQ_DAILY_URL = "https://stooq.com/q/d/l/"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/^GSPC"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
GDELT_DAILY_URL = "https://data.gdeltproject.org/events/{date}.export.CSV.zip"
SEC_LITIGATION_RSS_URL = "https://www.sec.gov/enforcement-litigation/litigation-releases/rss"
HTTP_RETRIES = 5
HTTP_RETRY_BASE_SECONDS = 2
HTTP_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

GDELT_COLUMNS = [
    "global_event_id", "sql_date", "month_year", "year", "fraction_date",
    "actor1_code", "actor1_name", "actor1_country", "actor1_known_group",
    "actor1_ethnic", "actor1_religion1", "actor1_religion2", "actor1_type1",
    "actor1_type2", "actor1_type3", "actor2_code", "actor2_name",
    "actor2_country", "actor2_known_group", "actor2_ethnic", "actor2_religion1",
    "actor2_religion2", "actor2_type1", "actor2_type2", "actor2_type3",
    "is_root_event", "event_code", "event_base_code", "event_root_code",
    "quad_class", "goldstein_scale", "num_mentions", "num_sources",
    "num_articles", "avg_tone", "actor1_geo_type", "actor1_geo_fullname",
    "actor1_geo_country", "actor1_geo_adm1", "actor1_geo_lat", "actor1_geo_long",
    "actor1_geo_feature_id", "actor2_geo_type", "actor2_geo_fullname",
    "actor2_geo_country", "actor2_geo_adm1", "actor2_geo_lat", "actor2_geo_long",
    "actor2_geo_feature_id", "action_geo_type", "action_geo_fullname",
    "action_geo_country", "action_geo_adm1", "action_geo_lat", "action_geo_long",
    "action_geo_feature_id", "date_added", "source_url",
]


def _get(url: str, *, timeout: int = 60, params: dict | None = None) -> requests.Response:
    """GET with bounded backoff for transient rate-limit/server failures."""
    last_response: requests.Response | None = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "market-predictor/0.1 (research ingestion)"},
            )
            last_response = response
            if response.status_code not in HTTP_RETRYABLE_STATUS or attempt >= HTTP_RETRIES:
                response.raise_for_status()
                return response
            retry_after = response.headers.get("Retry-After")
            try:
                delay = max(float(retry_after), 0.0) if retry_after else HTTP_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            except ValueError:
                delay = HTTP_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            print(
                f"HTTP retry {attempt}/{HTTP_RETRIES - 1}: status={response.status_code}; "
                f"waiting {delay:g}s before retry",
                flush=True,
            )
            time.sleep(delay)
        except requests.RequestException:
            if attempt >= HTTP_RETRIES:
                raise
            delay = HTTP_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            print(
                f"HTTP retry {attempt}/{HTTP_RETRIES - 1}: transient {type(last_response).__name__ if last_response is not None else 'request error'}; "
                f"waiting {delay:g}s before retry",
                flush=True,
            )
            time.sleep(delay)
    assert last_response is not None
    last_response.raise_for_status()
    return last_response


def _normalize_market_frame(frame: pd.DataFrame, *, source_id: str) -> pd.DataFrame:
    """Normalize an OHLCV frame and attach auditable source metadata."""
    frame = frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"market response missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("market response contains invalid dates")
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[required[1:]].isna().any().any():
        raise ValueError("market response contains non-numeric OHLCV values")
    raw_dates = frame["date"].dt.date
    invalid = [day.isoformat() for day in raw_dates if not _is_valid_session_date(day)]
    if invalid:
        raise ValueError(f"market response contains non-session dates: {invalid[:10]}")
    frame["date"] = [session_close(day) for day in raw_dates]
    frame = frame.set_index("date").sort_index()
    if frame.index.has_duplicates:
        raise ValueError("market response contains duplicate session dates")
    result = frame[["open", "high", "low", "close", "volume"]].astype(float)
    result.attrs["source_id"] = source_id
    return result
