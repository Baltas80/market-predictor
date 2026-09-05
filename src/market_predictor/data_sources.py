"""Adapters for reproducible real-world market, macro and event data.

The adapters deliberately keep source availability metadata separate from the
model features. They are ingestion primitives, not a claim that a source is
free of revisions or publication delays.
"""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
import zipfile

import pandas as pd
import requests


STOOQ_DAILY_URL = "https://stooq.com/q/d/l/"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
GDELT_DAILY_URL = "https://data.gdeltproject.org/events/{date}.export.CSV.zip"

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


def _get(url: str, *, timeout: int = 60) -> requests.Response:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "market-predictor/0.1"})
    response.raise_for_status()
    return response


def load_stooq_daily(symbol: str = "^spx", start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV data from Stooq and return a normalized frame."""
    params = {"s": symbol, "i": "d"}
    if start:
        params["d1"] = pd.Timestamp(start).strftime("%Y%m%d")
    if end:
        params["d2"] = pd.Timestamp(end).strftime("%Y%m%d")
    response = _get(STOOQ_DAILY_URL, timeout=60)
    # Stooq accepts query parameters; use a second request so parameters are explicit.
    response = requests.get(
        STOOQ_DAILY_URL,
        params=params,
        timeout=60,
        headers={"User-Agent": "market-predictor/0.1"},
    )
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), parse_dates=["Date"])
    frame.columns = [column.lower() for column in frame.columns]
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Stooq response missing columns: {sorted(missing)}")
    frame = frame.set_index("date").sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Stooq returned duplicate dates")
    return frame[list(required - {"date"})].astype(float)


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
    response = requests.get(
        FRED_OBSERVATIONS_URL,
        params=params,
        timeout=60,
        headers={"User-Agent": "market-predictor/0.1"},
    )
    response.raise_for_status()
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


def _gdelt_category(event_root_code: str, event_code: str, actor_text: str) -> str:
    """Conservative CAMEO-to-project mapping.

    The mapping intentionally uses only well-defined CAMEO families. Events
    that do not fit are assigned ``political_crisis`` rather than inventing a
    more specific label. Exact subcodes remain available in the raw frame.
    """
    root = str(event_root_code).zfill(2)
    code = str(event_code).zfill(4)
    actor_text = actor_text.lower()
    if code == "163" or code.startswith("163"):
        return "sanctions"
    if root == "14":
        return "social_unrest"
    if root in {"18", "19"}:
        return "war_conflict"
    if root == "20":
        if any(term in actor_text for term in ("terror", "militant", "extremist")):
            return "terrorism"
        return "war_conflict"
    if root in {"16", "17"}:
        return "political_crisis"
    if root in {"09", "10", "11", "12", "13", "15"}:
        return "political_crisis"
    return "political_crisis"


def load_gdelt_day(day: str | pd.Timestamp) -> pd.DataFrame:
    """Download one GDELT 2.0 daily event export and normalize key fields.

    GDELT's DATEADDED is retained as ``published_at`` because it is the
    machine-available timestamp. It is a conservative availability proxy, not
    a claim that the underlying article was first published at that instant.
    """
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
    })
    return output.drop_duplicates(subset=["event_id"]).sort_values("published_at")


def write_market_csv(frame: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True, index_label="date")


def write_events_csv(frame: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
