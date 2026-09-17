"""Canonical GDELT 1.0 daily-event loader with conservative PIT timing.

The ``events/`` archive is a daily batch. Its archive/event day is distinct
from information availability. DATEADDED is retained as original source data
only; PIT availability uses a conservative next-day 06:00 America/New_York
publication boundary.
"""
from __future__ import annotations

from datetime import timedelta, time as dt_time
from decimal import Decimal, InvalidOperation
from io import BytesIO
from zoneinfo import ZoneInfo
import zipfile

import pandas as pd
import requests

from .data_sources import _gdelt_category

GDELT_DAILY_URL = "https://data.gdeltproject.org/events/{date}.export.CSV.zip"
GDELT_SOURCE_ID = "GDELT_1_Event_Database"
GDELT_PUBLICATION_HOUR_EST = 6
GDELT_NEW_YORK = ZoneInfo("America/New_York")

GDELT_EXPORT_COLUMNS_61 = [
    "global_event_id", "sql_date", "month_year", "year", "fraction_date",
    "actor1_code", "actor1_name", "actor1_country", "actor1_known_group",
    "actor1_ethnic", "actor1_religion1", "actor1_religion2", "actor1_type1",
    "actor1_type2", "actor1_type3", "actor2_code", "actor2_name",
    "actor2_country", "actor2_known_group", "actor2_ethnic", "actor2_religion1",
    "actor2_religion2", "actor2_type1", "actor2_type2", "actor2_type3",
    "is_root_event", "event_code", "event_base_code", "event_root_code",
    "quad_class", "goldstein_scale", "num_mentions", "num_sources",
    "num_articles", "avg_tone", "actor1_geo_type", "actor1_geo_fullname",
    "actor1_geo_country", "actor1_geo_adm1", "actor1_geo_adm2",
    "actor1_geo_lat", "actor1_geo_long", "actor1_geo_feature_id",
    "actor2_geo_type", "actor2_geo_fullname", "actor2_geo_country",
    "actor2_geo_adm1", "actor2_geo_adm2", "actor2_geo_lat", "actor2_geo_long",
    "actor2_geo_feature_id", "action_geo_type", "action_geo_fullname",
    "action_geo_country", "action_geo_adm1", "action_geo_adm2",
    "action_geo_lat", "action_geo_long", "action_geo_feature_id",
    "date_added", "source_url",
]
GDELT_EXPORT_COLUMNS_58 = [
    column for column in GDELT_EXPORT_COLUMNS_61
    if column not in {"actor1_geo_adm2", "actor2_geo_adm2", "action_geo_adm2"}
]


def _canonical_gdelt_date_added(value: object) -> str | None:
    """Canonicalize original DATEADDED encodings without using them for PIT."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.isdigit() and len(text) == 14:
        return text
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite() or number != number.to_integral_value():
        return None
    canonical = format(number.to_integral_value(), "f")
    return canonical if canonical.isdigit() and len(canonical) == 14 else None


def _parse_gdelt_date_added(series: pd.Series) -> pd.Series:
    """Parse legacy 14-digit DATEADDED values as original source data only."""
    canonical = series.map(_canonical_gdelt_date_added).astype("string")
    return pd.to_datetime(canonical, format="%Y%m%d%H%M%S", utc=True, errors="coerce")


def gdelt1_daily_availability(day: str | pd.Timestamp) -> pd.Timestamp:
    """Return the conservative next-day 06:00 US Eastern PIT boundary."""
    archive_date = pd.Timestamp(day).date()
    local = pd.Timestamp.combine(
        archive_date + timedelta(days=1),
        dt_time(hour=GDELT_PUBLICATION_HOUR_EST),
    ).tz_localize(GDELT_NEW_YORK)
    return local.tz_convert("UTC")


def load_gdelt_day(day: str | pd.Timestamp) -> pd.DataFrame:
    """Load one GDELT 1.0 daily archive without using DATEADDED as PIT."""
    date = pd.Timestamp(day).strftime("%Y%m%d")
    response = requests.get(
        GDELT_DAILY_URL.format(date=date),
        timeout=120,
        headers={"User-Agent": "market-predictor/0.1 (research ingestion)"},
    )
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        members = archive.namelist()
        if not members:
            raise ValueError(f"Empty GDELT archive for {date}")
        member = members[0]
        with archive.open(member) as probe:
            field_count = probe.readline().count(b"\t") + 1
        if field_count == len(GDELT_EXPORT_COLUMNS_61):
            columns = GDELT_EXPORT_COLUMNS_61
        elif field_count == len(GDELT_EXPORT_COLUMNS_58):
            columns = GDELT_EXPORT_COLUMNS_58
        else:
            raise ValueError(f"Unsupported GDELT field count for {date}: {field_count} (expected 58 or 61)")
        with archive.open(member) as handle:
            frame = pd.read_csv(handle, sep="\t", header=None, names=columns, dtype=str, low_memory=False)

    for column in ("actor1_geo_adm2", "actor2_geo_adm2", "action_geo_adm2"):
        if column not in frame:
            frame[column] = pd.NA
    frame = frame[GDELT_EXPORT_COLUMNS_61]
    frame["date_added"] = frame["date_added"].astype("string").str.strip()
    frame["sql_date"] = pd.to_datetime(frame["sql_date"].astype("string").str.strip(), format="%Y%m%d", utc=True, errors="coerce")
    for column in ["goldstein_scale", "num_mentions", "num_sources", "num_articles", "avg_tone"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    actor_text = frame[["actor1_name", "actor2_name", "action_geo_fullname"]].fillna("").agg(" ".join, axis=1)
    frame["category"] = [_gdelt_category(root, code, text) for root, code, text in zip(frame["event_root_code"], frame["event_code"], actor_text)]
    scale = frame["goldstein_scale"].abs().clip(0, 10) / 10
    media = frame["num_sources"].fillna(0).clip(lower=0)
    frame["severity"] = (0.7 * scale + 0.3 * (media / (media + 5)).clip(0, 1)).clip(0, 1)
    frame["surprise"] = 0.0
    frame["available_at"] = gdelt1_daily_availability(day)
    frame["source_id"] = GDELT_SOURCE_ID
    return frame
