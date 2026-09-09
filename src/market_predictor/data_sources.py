"""Adapters for reproducible real-world market, macro and event data."""

from __future__ import annotations

from datetime import timedelta
from io import BytesIO, StringIO
from pathlib import Path
import hashlib
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
    response = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "market-predictor/0.1 (research ingestion)"},
    )
    response.raise_for_status()
    return response


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


def _load_yahoo_spx_daily(start: str | None, end: str | None) -> pd.DataFrame:
    """Fallback OHLCV retrieval for S&P 500 when Stooq is unavailable."""
    start_ts = pd.Timestamp(start or "2000-01-01", tz="UTC")
    end_ts = pd.Timestamp(end or pd.Timestamp.utcnow().date(), tz="UTC") + pd.Timedelta(days=1)
    params = {
        "period1": int(start_ts.timestamp()),
        "period2": int(end_ts.timestamp()),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "false",
    }
    response = _get(YAHOO_CHART_URL, timeout=60, params=params)
    payload = response.json()
    result = payload.get("chart", {}).get("result") or []
    if not result:
        error = payload.get("chart", {}).get("error")
        raise ValueError(f"Yahoo Finance returned no chart data: {error}")
    chart = result[0]
    timestamps = chart.get("timestamp") or []
    quotes = (chart.get("indicators", {}).get("quote") or [{}])[0]
    if not timestamps or not quotes:
        raise ValueError("Yahoo Finance returned an empty S&P 500 quote series")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True),
            "open": quotes.get("open", []),
            "high": quotes.get("high", []),
            "low": quotes.get("low", []),
            "close": quotes.get("close", []),
            "volume": quotes.get("volume", []),
        }
    )
    frame["volume"] = frame["volume"].fillna(0)
    return _normalize_market_frame(frame, source_id="YahooFinance_GSPC")


def load_stooq_daily(symbol: str = "^spx", start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Load daily bars, falling back to Yahoo's S&P 500 chart when Stooq is unavailable."""
    params = {"s": symbol, "i": "d"}
    if start:
        params["d1"] = pd.Timestamp(start).strftime("%Y%m%d")
    if end:
        params["d2"] = pd.Timestamp(end).strftime("%Y%m%d")
    try:
        response = _get(STOOQ_DAILY_URL, timeout=60, params=params)
        frame = pd.read_csv(StringIO(response.text))
        return _normalize_market_frame(frame, source_id="Stooq_SPX")
    except (ValueError, pd.errors.ParserError, KeyError) as stooq_error:
        fallback = _load_yahoo_spx_daily(start, end)
        fallback.attrs["market_source_fallback"] = "Stooq unavailable or returned invalid OHLCV response"
        fallback.attrs["market_source_error"] = str(stooq_error)
        return fallback


def _is_valid_session_date(day) -> bool:
    try:
        session_close(day)
        return True
    except ValueError:
        return False


def load_fred_observations(
    series_id: str,
    api_key: str,
    *,
    realtime_start: str | None = None,
    realtime_end: str | None = None,
) -> pd.DataFrame:
    """Load FRED observations in tall real-time-period format.

    FRED ``output_type=2`` is a wide vintage matrix with dynamically named
    columns (for example ``CPIAUCSL_20200101``), so it cannot be represented
    by the point-in-time audit schema. Output type 1 preserves the per-row
    ``realtime_start``/``realtime_end`` fields needed by the PIT pipeline.
    Empty observations are represented by FRED as non-numeric values such as
    ``.`` and are removed before the PIT audit because they are not actual
    observations.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "output_type": 1,
    }
    if realtime_start:
        params["realtime_start"] = realtime_start
    if realtime_end:
        params["realtime_end"] = realtime_end
    response = _get(FRED_OBSERVATIONS_URL, timeout=60, params=params)
    frame = pd.DataFrame(response.json().get("observations", []))
    if frame.empty:
        return pd.DataFrame(columns=["series_id", "date", "value", "realtime_start", "realtime_end"])
    frame["series_id"] = series_id
    required = ["series_id", "date", "value", "realtime_start", "realtime_end"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"FRED observations missing columns: {sorted(missing)}")
    for column in ("date", "realtime_start", "realtime_end"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.loc[frame["value"].notna()].copy()
    audit_fred_point_in_time(frame[required])
    return frame.dropna(subset=["date", "value"]).sort_values(["date", "realtime_start"])


def align_fred_point_in_time(observations: pd.DataFrame, market_index: pd.DatetimeIndex, *, conservative_session_lag: int = 1) -> pd.DataFrame:
    required = {"date", "value", "realtime_start"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"FRED observations missing columns: {sorted(missing)}")
    if conservative_session_lag < 0:
        raise ValueError("conservative_session_lag must be >= 0")
    if {"series_id", "realtime_end"}.issubset(observations.columns):
        audit_fred_point_in_time(observations[["series_id", "date", "value", "realtime_start", "realtime_end"]])
    market = pd.DatetimeIndex(market_index).tz_localize(None).normalize()
    vintages = observations.copy()
    vintages["date"] = pd.to_datetime(vintages["date"], errors="coerce").dt.tz_localize(None)
    vintages["realtime_start"] = pd.to_datetime(vintages["realtime_start"], errors="coerce").dt.tz_localize(None)
    vintages = vintages.dropna(subset=["date", "realtime_start", "value"])
    rows = []
    for timestamp in market:
        cutoff = timestamp - pd.Timedelta(value=int(conservative_session_lag), unit="D")
        eligible = vintages[vintages["realtime_start"] <= cutoff]
        if eligible.empty:
            rows.append({"date": timestamp, "value": pd.NA, "vintage": pd.NaT})
            continue
        latest = eligible.sort_values("realtime_start").drop_duplicates("date", keep="last")
        latest = latest[latest["date"] <= timestamp]
        if latest.empty:
            rows.append({"date": timestamp, "value": pd.NA, "vintage": pd.NaT})
            continue
        row = latest.sort_values("date").iloc[-1]
        rows.append({"date": timestamp, "value": row["value"], "vintage": row["realtime_start"]})
    return pd.DataFrame(rows).set_index("date")


def _gdelt_category(event_root_code: str, event_code: str, actor_text: str) -> str:
    root = str(event_root_code).zfill(2)
    code = str(event_code).zfill(4)
    actor_text = actor_text.lower()
    if code == "0163" or code.startswith("0163"):
        return "sanctions"
    if root == "14":
        return "social_unrest"
    if root in {"18", "19", "20"}:
        return "war_conflict" if root != "20" or not any(t in actor_text for t in ("terror", "militant", "extremist")) else "terrorism"
    if root in {"16", "17", "09", "10", "11", "12", "13", "15"}:
        return "political_crisis"
    return "political_crisis"


def load_gdelt_day(day: str | pd.Timestamp) -> pd.DataFrame:
    date = pd.Timestamp(day).strftime("%Y%m%d")
    response = _get(GDELT_DAILY_URL.format(date=date), timeout=120)
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        members = archive.namelist()
        if not members:
            raise ValueError(f"Empty GDELT archive for {date}")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(handle, sep="\t", header=None, names=GDELT_COLUMNS, dtype=str, low_memory=False)
    frame["date_added"] = pd.to_datetime(frame["date_added"], format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    frame["sql_date"] = pd.to_datetime(frame["sql_date"], format="%Y%m%d", errors="coerce", utc=True)
    for column in ["goldstein_scale", "num_mentions", "num_sources", "num_articles", "avg_tone"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    actor_text = frame[["actor1_name", "actor2_name", "action_geo_fullname"]].fillna("").agg(" ".join, axis=1)
    frame["category"] = [_gdelt_category(root, code, text) for root, code, text in zip(frame["event_root_code"], frame["event_code"], actor_text)]
    scale = frame["goldstein_scale"].abs().clip(0, 10) / 10
    media = frame["num_sources"].fillna(0).clip(lower=0)
    frame["severity"] = (0.7 * scale + 0.3 * (media / (media + 5)).clip(0, 1)).clip(0, 1)
    frame["surprise"] = 0.0
    return frame


def load_gdelt_range(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    start_date, end_date = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    if end_date < start_date:
        raise ValueError("end must be on or after start")
    frames = []
    current = start_date
    while current <= end_date:
        frames.append(load_gdelt_day(current))
        current += timedelta(days=1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=GDELT_COLUMNS + ["category", "severity", "surprise"])


def _sec_category(title: str, description: str) -> str:
    text = f"{title} {description}".lower()
    if any(t in text for t in ("fraud", "accounting", "ponzi", "misstatement", "financial reporting")):
        return "financial_fraud"
    if any(t in text for t in ("insider trading", "bribery", "corruption", "scandal", "false statements")):
        return "corporate_scandal"
    return "regulation"


def load_sec_litigation_releases_rss() -> pd.DataFrame:
    response = _get(SEC_LITIGATION_RSS_URL, timeout=60)
    root = ET.fromstring(response.content)
    rows = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        description = (item.findtext("description") or "").strip()
        published = (item.findtext("pubDate") or "").strip()
        link = (item.findtext("link") or "").strip()
        guid = (item.findtext("guid") or link or title).strip()
        published_at = pd.to_datetime(published, utc=True, errors="coerce")
        if pd.isna(published_at):
            continue
        rows.append({"event_id": f"sec:{hashlib.sha256(guid.encode()).hexdigest()[:24]}", "category": _sec_category(title, description), "published_at": published_at, "available_at": published_at, "event_time": published_at, "severity": 0.5, "country": "US", "entity": title, "sector": pd.NA, "duration_days": 0.0, "media_intensity": 0.0, "surprise": 0.0, "source_url": link, "source": "SEC_Litigation_Releases"})
    return pd.DataFrame(rows).drop_duplicates("event_id").sort_values("published_at")


def gdelt_events_to_market_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Map GDELT without conflating occurrence, publication and availability."""
    event_time = pd.to_datetime(frame["sql_date"], utc=True, errors="coerce")
    available_at = pd.to_datetime(frame["date_added"], utc=True, errors="coerce")
    return pd.DataFrame({
        "event_id": frame["global_event_id"].astype(str), "event_time": event_time,
        "published_at": pd.NaT, "available_at": available_at, "category": frame["category"],
        "severity": frame["severity"].round(6), "country": frame["action_geo_country"].replace("", pd.NA),
        "entity": frame["actor2_name"].replace("", pd.NA), "sector": pd.NA, "duration_days": 0.0,
        "media_intensity": frame["num_sources"].fillna(0), "surprise": frame["surprise"],
        "source": "GDELT_2_Event_Database", "availability_proxy": "DATEADDED", "source_url": frame["source_url"],
    }).drop_duplicates("event_id").sort_values("available_at")


def write_market_csv(frame: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True, index_label="date")


def write_events_csv(frame: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
