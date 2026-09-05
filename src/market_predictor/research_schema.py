"""Definitive point-in-time schema for market, macro and event research data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


MARKET_COLUMNS = ("open", "high", "low", "close", "volume")
MACRO_COLUMNS = ("series_id", "observation_date", "value", "vintage_start", "vintage_end")
EVENT_COLUMNS = (
    "event_id", "event_time", "published_at", "available_at", "source_id",
    "category", "severity", "country", "entity", "sector", "duration_days",
    "media_intensity", "surprise",
)


def _utc_timestamp(value: object, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid timestamp")
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def validate_market_frame(frame: pd.DataFrame) -> None:
    missing = set(MARKET_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"market data missing columns: {sorted(missing)}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise TypeError("market index must be a timezone-aware DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("market index must be unique and chronological")
    numeric = frame.loc[:, list(MARKET_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("market OHLCV must be finite numeric values")
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("market prices must be positive")
    if (numeric["volume"] < 0).any():
        raise ValueError("market volume must be nonnegative")
    if (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high must be >= open, close and low")
    if (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low must be <= open, close and high")


def validate_macro_frame(frame: pd.DataFrame) -> None:
    missing = set(MACRO_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"macro data missing columns: {sorted(missing)}")
    data = frame.copy()
    for column in ("observation_date", "vintage_start", "vintage_end"):
        data[column] = pd.to_datetime(data[column], utc=True, errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    if data[list(MACRO_COLUMNS)].isna().any().any():
        raise ValueError("macro schema contains invalid or missing values")
    if not np.isfinite(data["value"].to_numpy(dtype=float)).all():
        raise ValueError("macro values must be finite")
    if (data["vintage_start"] < data["observation_date"].dt.normalize()).any():
        raise ValueError("macro vintage_start cannot precede observation_date")
    if (data["vintage_end"] < data["vintage_start"]).any():
        raise ValueError("macro vintage_end cannot precede vintage_start")
    if data.duplicated(["series_id", "observation_date", "vintage_start"]).any():
        raise ValueError("duplicate macro vintage keys")


def validate_event_frame(frame: pd.DataFrame) -> None:
    missing = set(EVENT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"event data missing columns: {sorted(missing)}")
    data = frame.copy()
    for column in ("event_time", "published_at", "available_at"):
        data[column] = pd.to_datetime(data[column], utc=True, errors="coerce")
    data["severity"] = pd.to_numeric(data["severity"], errors="coerce")
    required = ["event_id", "source_id", "category", "event_time", "available_at", "severity"]
    if data[required].isna().any().any():
        raise ValueError("event identifiers, event_time, available_at and severity are required")
    if (data["available_at"] < data["event_time"]).any():
        raise ValueError("available_at cannot precede event_time")
    published = data["published_at"].notna()
    if (data.loc[published, "available_at"] < data.loc[published, "published_at"]).any():
        raise ValueError("available_at cannot precede published_at")
    if (data.loc[published, "published_at"] < data.loc[published, "event_time"]).any():
        raise ValueError("published_at cannot precede event_time")
    if (data["severity"] < 0).any() or (data["severity"] > 1).any():
        raise ValueError("event severity must be between 0 and 1")
    if data["event_id"].duplicated().any():
        raise ValueError("event_id must be unique after deduplication")


def normalize_event_sources(frame: pd.DataFrame, *, source_id: str) -> pd.DataFrame:
    """Normalize source-specific event columns into the definitive event schema."""
    data = frame.copy()
    if "event_id" not in data:
        raise ValueError("source event data requires event_id")
    if "event_time" not in data:
        if "date" in data:
            data["event_time"] = data["date"]
        elif "sql_date" in data:
            data["event_time"] = data["sql_date"]
        else:
            data["event_time"] = data.get("published_at")
    if "published_at" not in data:
        data["published_at"] = pd.NaT
    if "available_at" not in data:
        data["available_at"] = data["published_at"]
    data["source_id"] = source_id
    defaults = {
        "category": "political_crisis", "severity": 0.0, "country": pd.NA,
        "entity": pd.NA, "sector": pd.NA, "duration_days": 0.0,
        "media_intensity": 0.0, "surprise": 0.0,
    }
    for column, default in defaults.items():
        if column not in data:
            data[column] = default
    output = data[list(EVENT_COLUMNS)].copy()
    for column in ("event_time", "published_at", "available_at"):
        output[column] = pd.to_datetime(output[column], utc=True, errors="coerce")
    return output


def deduplicate_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by event_id, retaining the earliest available record."""
    data = frame.copy()
    if data["event_id"].duplicated().any():
        data = data.sort_values(["event_id", "available_at", "published_at"], na_position="last")
        data = data.groupby("event_id", sort=False, as_index=False).first()
    data = data.sort_values(["available_at", "event_id"]).reset_index(drop=True)
    validate_event_frame(data)
    return data


def admitted_events(events: pd.DataFrame, decision_time: datetime | str) -> pd.DataFrame:
    """Return only events whose information was available by a decision time."""
    validate_event_frame(events)
    cutoff = _utc_timestamp(decision_time, "decision_time")
    data = events.copy()
    data["available_at"] = pd.to_datetime(data["available_at"], utc=True)
    return data.loc[data["available_at"] <= cutoff].copy()


@dataclass(frozen=True)
class HistoricalSchema:
    """Versioned schema identity used by dataset manifests and lockboxes."""

    version: str = "2026-09-05"
    market_time: str = "decision_time"
    macro_time: str = "observation_date+vintage_start"
    event_times: tuple[str, str, str] = ("event_time", "published_at", "available_at")

    def validate(self) -> None:
        if not self.version or len(self.event_times) != 3:
            raise ValueError("invalid historical schema identity")
