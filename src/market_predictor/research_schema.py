"""Definitive point-in-time schema for market, macro and event research data.

The schema makes three different notions of time explicit:
- event_time: when the underlying event happened;
- published_at: when the source published the information;
- available_at: when the observation is admitted to the model information set.

Keeping these timestamps separate prevents publication/availability time from
being accidentally inferred from the event date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

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
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("market index must be a DatetimeIndex")
    if frame.index.tz is None:
        raise ValueError("market index must be timezone-aware")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("market index must be unique and chronological")
    numeric = frame.loc[:, list(MARKET_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if not numeric.map(lambda column: column.notna().all() and column.map(pd.api.types.is_number).all()).all():
        raise ValueError("market OHLCV must be numeric and finite")
    if not numeric.apply(lambda column: column.map(pd.notna).all()).all():
        raise ValueError("market OHLCV cannot contain missing values")
    if not numeric.apply(lambda column: column.map(pd.api.types.is_number).all()).all():
        raise ValueError("market OHLCV must be numeric")
    if not numeric.apply(lambda column: column.map(lambda value: pd.notna(value) and float(value) == float(value) and abs(float(value)) != float('inf')).all()).all():
        raise ValueError("market OHLCV must be finite")
    if (numeric["close"] <= 0).any() or (numeric["open"] <= 0).any() or (numeric["high"] <= 0).any() or (numeric["low"] <= 0).any():
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
    if (data["vintage_start"] < data["observation_date"].dt.normalize()).any():
        raise ValueError("macro vintage_start cannot precede observation_date")
    if (data["vintage_end"] < data["vintage_start"]).any():
        raise ValueError("macro vintage_end cannot precede vintage_start")
    keys = ["series_id", "observation_date", "vintage_start"]
    if data.duplicated(keys).any():
        raise ValueError("duplicate macro vintage keys")


def validate_event_frame(frame: pd.DataFrame) -> None:
    missing = set(EVENT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"event data missing columns: {sorted(missing)}")
    data = frame.copy()
    for column in ("event_time", "published_at", "available_at"):
        data[column] = pd.to_datetime(data[column], utc=True, errors="coerce")
    data["severity"] = pd.to_numeric(data["severity"], errors="coerce")
    if data[["event_id", "source_id", "category"]].isna().any().any():
        raise ValueError("event identifiers and category are required")
    if data[["event_time", "published_at", "available_at", "severity"]].isna().any().any():
        raise ValueError("event timestamps and severity must be valid")
    if (data["available_at"] < data["published_at"]).any():
        raise ValueError("available_at cannot precede published_at")
    if (data["severity"] < 0).any() or (data["severity"] > 1).any():
        raise ValueError("event severity must be between 0 and 1")
    if data["event_id"].duplicated().any():
        raise ValueError("event_id must be unique after deduplication")


def deduplicate_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by source identity while retaining the earliest availability."""
    validate_event_frame(frame)
    data = frame.copy()
    for column in ("event_time", "published_at", "available_at"):
        data[column] = pd.to_datetime(data[column], utc=True)
    data = data.sort_values(["event_id", "available_at", "published_at"])
    if data["event_id"].duplicated().any():
        first = data.groupby("event_id", sort=False, as_index=False).first()
        first = first.sort_values("available_at").reset_index(drop=True)
        validate_event_frame(first)
        return first
    return data.reset_index(drop=True)


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
