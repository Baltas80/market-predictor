"""Deterministic data-quality checks for market, macro and event inputs."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityCheck:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_market(frame: pd.DataFrame) -> list[QualityCheck]:
    checks: list[QualityCheck] = []
    required = ["open", "high", "low", "close", "volume"]
    checks.append(QualityCheck("market_columns", set(required).issubset(frame.columns), "required OHLCV columns present"))
    if not set(required).issubset(frame.columns):
        return checks
    idx = pd.DatetimeIndex(frame.index)
    checks.append(QualityCheck("market_index_datetime", isinstance(frame.index, pd.DatetimeIndex), "index must be a DatetimeIndex"))
    checks.append(QualityCheck("market_index_unique", not idx.has_duplicates, "dates must be unique"))
    checks.append(QualityCheck("market_index_sorted", idx.is_monotonic_increasing, "dates must be chronological"))
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    checks.append(QualityCheck("market_numeric", not numeric.isna().any().any(), "OHLCV must be numeric and non-null"))
    checks.append(QualityCheck("market_finite", bool(np.isfinite(numeric.to_numpy(dtype=float)).all()), "OHLCV must be finite"))
    prices = numeric[["open", "high", "low", "close"]]
    checks.append(QualityCheck("market_prices_positive", bool((prices > 0).all().all()), "prices must be strictly positive"))
    checks.append(QualityCheck("market_volume_nonnegative", bool((numeric["volume"] >= 0).all()), "volume cannot be negative"))
    checks.append(QualityCheck("market_ohlc_bounds", bool(((numeric.high >= numeric.low) & (numeric.open.between(numeric.low, numeric.high)) & (numeric.close.between(numeric.low, numeric.high))).all()), "high >= low and open/close inside the daily range"))
    return checks


def check_macro(frame: pd.DataFrame) -> list[QualityCheck]:
    checks: list[QualityCheck] = []
    required = {"date", "value", "realtime_start"}
    checks.append(QualityCheck("macro_columns", required.issubset(frame.columns), "required FRED vintage columns present"))
    if required.issubset(frame.columns):
        date = pd.to_datetime(frame["date"], errors="coerce", utc=True)
        start = pd.to_datetime(frame["realtime_start"], errors="coerce", utc=True)
        end = pd.to_datetime(frame["realtime_end"], errors="coerce", utc=True) if "realtime_end" in frame else start
        checks.append(QualityCheck("macro_timestamps_valid", not date.isna().any() and not start.isna().any() and not end.isna().any(), "all timestamps must parse"))
        checks.append(QualityCheck("macro_vintage_interval", bool((end >= start).all()), "realtime_end must not precede realtime_start"))
        checks.append(QualityCheck("macro_value_numeric", pd.to_numeric(frame["value"], errors="coerce").notna().all(), "macro values must be numeric"))
        checks.append(QualityCheck("macro_vintage_unique", not frame.duplicated(["date", "realtime_start"]).any(), "duplicate observation/vintage pairs are forbidden"))
    return checks


def check_events(frame: pd.DataFrame) -> list[QualityCheck]:
    checks: list[QualityCheck] = []
    required = {"event_id", "published_at", "severity"}
    checks.append(QualityCheck("event_columns", required.issubset(frame.columns), "required event columns present"))
    if required.issubset(frame.columns):
        published = pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
        severity = pd.to_numeric(frame["severity"], errors="coerce")
        checks.append(QualityCheck("event_timestamps_valid", not published.isna().any(), "published_at must parse"))
        checks.append(QualityCheck("event_ids_unique", not frame["event_id"].duplicated().any(), "event_id must be unique after deduplication"))
        checks.append(QualityCheck("event_severity_range", bool(severity.between(0, 1).all()), "severity must be in [0, 1]"))
        if "duration_days" in frame:
            duration = pd.to_numeric(frame["duration_days"], errors="coerce")
            checks.append(QualityCheck("event_duration_nonnegative", bool(duration.dropna().ge(0).all()), "duration_days cannot be negative"))
        if "media_intensity" in frame:
            media = pd.to_numeric(frame["media_intensity"], errors="coerce")
            checks.append(QualityCheck("event_media_nonnegative", bool(media.dropna().ge(0).all()), "media_intensity cannot be negative"))
    return checks


def quality_report(*, market: pd.DataFrame | None = None, macro: pd.DataFrame | None = None, events: pd.DataFrame | None = None) -> dict[str, Any]:
    """Return a machine-readable report; no input is silently repaired."""
    checks: list[QualityCheck] = []
    if market is not None:
        checks.extend(check_market(market))
    if macro is not None:
        checks.extend(check_macro(macro))
    if events is not None:
        checks.extend(check_events(events))
    return {"passed": all(check.passed for check in checks), "checks": [check.as_dict() for check in checks]}
