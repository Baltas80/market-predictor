"""Adapters for reproducible real-world market, macro and event data.

The adapters deliberately keep source availability metadata separate from the
model features. They are ingestion primitives, not a claim that a source is
free of revisions or publication delays.
"""

from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO, StringIO
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import requests


STOOQ_DAILY_URL = "https://stooq.com/q/d/l/"
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


def load_stooq_daily(symbol: str = "^spx", start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV data from Stooq and return a normalized frame."""
    params = {"s": symbol, "i": "d"}
    if start:
        params["d1"] = pd.Timestamp(start).strftime("%Y%m%d")
    if end:
        params["d2"] = pd.Timestamp(end).strftime("%Y%m%d")
    response = _get(STOOQ_DAILY_URL, timeout=60, params=params)
    frame = pd.read_csv(StringIO(response.text), parse_dates=["Date"])
    frame.columns = [column.lower() for column in frame.columns]
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Stooq response missing columns: {sorted(missing)}")
    frame = frame.set_index("date").sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Stooq returned duplicate dates")
    return frame[["open", "high", "low", "close", "volume"]].astype(float)


def load_fred_observations(
    series_id: str,
    api_key: str,
    *,
    realtime_start: str | None = None,
    realtime_end: str | None = None,
) -> pd.DataFrame:
    """Load FRED observations with real-time/vintage metadata preserved."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "output_type": 2,
    }
    if realtime_start:
        params["realtime_start"] = realtime_start
    if realtime_end:
        params["realtime_end"] = realtime_end
    response = _get(FRED_OBSERVATIONS_URL, timeout=60, params=params)
    payload = response.json()
    observations = payload.get("observations", [])
    frame = pd.DataFrame(observations)
    if frame.empty:
        return pd.DataFrame(columns=["date", "value", "realtime_start", "realtime_end"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["realtime_start"] = pd.to_datetime(frame["realtime_start"], errors="coerce")
    frame["realtime_end"] = pd.to_datetime(frame["realtime_end"], errors="coerce")
    return frame.dropna(subset=["date", "value"]).sort_values(["date", "realtime_start"])


def align_fred_point_in_time(
    observations: pd.DataFrame,
    market_index: pd.DatetimeIndex,
    *,
    conservative_session_lag: int = 1,
) -> pd.DataFrame:
    """Align FRED vintages without using revisions that were not yet known.

    FRED vintage dates are date-level availability markers, not intraday
    release timestamps. The default lag is therefore expressed in calendar
    days and is deliberately conservative until exact release times are added.
    """
    required = {"date", "value", "realtime_start"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"FRED observations missing columns: {sorted(missing)}")
    if conservative_session_lag < 0:
        raise ValueError("conservative_session_lag must be >= 0")
    market = pd.DatetimeIndex(market_index).tz_localize(None).normalize()
    vintages = observations.copy()
    vintages["date"] = pd.to_datetime(vintages["date"], errors="coerce").dt.tz_localize(None)
    vintages["realtime_start"] = pd.to_datetime(vintages["realtime_start"], errors="coerce").dt.tz_localize(None)
    vintages = vintages.dropna(subset=["date", "realtime_start", "value"])
    rows: list[dict[str, object]] = []
    for timestamp in market:
        cutoff = timestamp - pd.Timedelta(days=conservative_session_lag)
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
    """Conservative CAMEO-to-project mapping; raw codes remain available."""
    root = str(event_root_code).zfill(2)
    code = str(event_code).zfill(4)
    actor_text = actor_text.lower()
    if code == "0163" or code.startswith("0163"):
        return "sanctions"
    if root == "14":
        return "social_unrest"
    if root in {"18", "19"}:
        return "war_conflict"
    if root == "20":
        if any(term in actor_text for term in ("terror", "militant", "extremist")):
            return "terrorism"
        return "war_conflict"
    if root in {"16", "17", "09", "10", "11", "12", "13", "15"}:
        return "political_crisis"
    return "political_crisis"


def load_gdelt_day(day: str | pd.Timestamp) -> pd.DataFrame:
    """Download one GDELT 2.0 daily event export and normalize key fields."""
    date = pd.Timestamp(day).strftime("%Y%m%d")
    response = _get(GDELT_DAILY_URL.format(date=date), timeout=120)
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        members = archive.namelist()
        if not members:
            raise ValueError(f"Empty GDELT archive for {date}")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(
                handle,
                sep="\t",
                header=None,
                names=GDELT_COLUMNS,
                dtype=str,
                low_memory=False,
            )
    frame["date_added"] = pd.to_datetime(frame["date_added"], format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    frame["sql_date"] = pd.to_datetime(frame["sql_date"], format="%Y%m%d", errors="coerce")
    for column in ["goldstein_scale", "num_mentions", "num_sources", "num_articles", "avg_tone"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    actor_text = frame[["actor1_name", "actor2_name", "action_geo_fullname"]].fillna("").agg(" ".join, axis=1)
    frame["category"] = [
        _gdelt_category(root, code, text)
        for root, code, text in zip(frame["event_root_code"], frame["event_code"], actor_text)
    ]
    scale = frame["goldstein_scale"].abs().clip(0, 10) / 10
    media = frame["num_sources"].fillna(0).clip(lower=0)
    media_norm = (media / (media + 5)).clip(0, 1)
    frame["severity"] = (0.7 * scale + 0.3 * media_norm).clip(0, 1)
    frame["surprise"] = 0.0
    return frame


def load_gdelt_range(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Download GDELT daily files for an inclusive range and concatenate them.

    Failed or missing dates raise an exception rather than silently producing
    an incomplete historical corpus. Callers can resume by splitting the range.
    """
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    if end_date < start_date:
        raise ValueError("end must be on or after start")
    frames = []
    current = start_date
    while current <= end_date:
        frames.append(load_gdelt_day(current))
        current += timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=GDELT_COLUMNS + ["category", "severity", "surprise"])
    return pd.concat(frames, ignore_index=True)


def _sec_category(title: str, description: str) -> str:
    text = f"{title} {description}".lower()
    fraud_terms = ("fraud", "accounting", "ponzi", "misstatement", "financial reporting")
    scandal_terms = ("insider trading", "bribery", "corruption", "scandal", "false statements")
    if any(term in text for term in fraud_terms):
        return "financial_fraud"
    if any(term in text for term in scandal_terms):
        return "corporate_scandal"
    return "regulation"


def load_sec_litigation_releases_rss() -> pd.DataFrame:
    """Load SEC Litigation Releases from the official RSS feed.

    The SEC feed provides publication dates, titles, links and summaries. The
    publication date is retained as ``published_at`` and is treated as the
    information-availability timestamp for a daily model.
    """
    response = _get(SEC_LITIGATION_RSS_URL, timeout=60)
    root = ET.fromstring(response.content)
    rows: list[dict[str, object]] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        description = (item.findtext("description") or "").strip()
        published = (item.findtext("pubDate") or "").strip()
        link = (item.findtext("link") or "").strip()
        guid = (item.findtext("guid") or link or title).strip()
        published_at = pd.to_datetime(published, utc=True, errors="coerce")
        if pd.isna(published_at):
            continue
        rows.append(
            {
                "event_id": f"sec:{hashlib.sha256(guid.encode('utf-8')).hexdigest()[:24]}",
                "category": _sec_category(title, description),
                "published_at": published_at.isoformat().replace("+00:00", "Z"),
                "severity": 0.5,
                "country": "US",
                "entity": title,
                "sector": pd.NA,
                "duration_days": 0.0,
                "media_intensity": 0.0,
                "surprise": 0.0,
                "source_url": link,
                "source": "SEC_Litigation_Releases",
            }
        )
    return pd.DataFrame(rows).drop_duplicates("event_id").sort_values("published_at")


def gdelt_events_to_market_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert normalized GDELT rows to the project's event CSV schema."""
    output = pd.DataFrame({
        "event_id": frame["global_event_id"].astype(str),
        "category": frame["category"],
        "published_at": frame["date_added"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "severity": frame["severity"].round(6),
        "country": frame["action_geo_country"].replace("", pd.NA),
        "entity": frame["actor2_name"].replace("", pd.NA),
        "sector": pd.NA,
        "duration_days": 0.0,
        "media_intensity": frame["num_sources"].fillna(0),
        "surprise": frame["surprise"],
        "source": "GDELT_2_Event_Database",
        "availability_proxy": "DATEADDED",
    })
    return output.drop_duplicates(subset=["event_id"]).sort_values("published_at")


def write_market_csv(frame: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True, index_label="date")


def write_events_csv(frame: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
