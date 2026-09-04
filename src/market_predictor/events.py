"""Point-in-time geopolitical and macro event feature engineering.

The event layer is deliberately separated from data acquisition. A normalized
CSV/Parquet source can be converted into daily features without allowing an
event to influence a prediction before that event was observable.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_EVENT_COLUMNS = {
    "event_time",
    "available_time",
    "event_type",
    "intensity",
}

DEFAULT_WINDOWS = (1, 3, 5, 10, 20)


def validate_events(events: pd.DataFrame) -> None:
    """Validate the minimum point-in-time event schema."""
    missing = REQUIRED_EVENT_COLUMNS - set(events.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")

    event_time = pd.to_datetime(events["event_time"], utc=True, errors="coerce")
    available_time = pd.to_datetime(events["available_time"], utc=True, errors="coerce")
    if event_time.isna().any() or available_time.isna().any():
        raise ValueError("event_time and available_time must contain valid timestamps")
    if (available_time < event_time).any():
        raise ValueError("available_time cannot precede event_time")


def _daily_event_table(events: pd.DataFrame) -> pd.DataFrame:
    validate_events(events)
    out = events.copy()
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True)
    out["available_time"] = pd.to_datetime(out["available_time"], utc=True)
    out["event_date"] = out["event_time"].dt.floor("D")
    out["available_date"] = out["available_time"].dt.floor("D")
    out["intensity"] = pd.to_numeric(out["intensity"], errors="coerce").fillna(0.0)
    out["is_conflict"] = out["event_type"].astype(str).str.lower().isin(
        {"war", "armed_conflict", "military_attack", "terrorism", "civil_unrest", "sanction"}
    ).astype(int)
    return out


def build_event_features(
    events: pd.DataFrame,
    dates: Iterable[pd.Timestamp],
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Create leakage-safe daily event features aligned to prediction dates.

    ``available_time`` is used rather than only ``event_time``. If a historical
    event was published/observed after a market observation, it cannot affect
    that observation's features.

    The output contains counts, conflict counts, mean/max intensity and
    exponentially decaying event pressure for each requested window.
    """
    daily = _daily_event_table(events)
    index = pd.DatetimeIndex(pd.to_datetime(list(dates), utc=True)).floor("D")
    index = index.sort_values().unique()
    result = pd.DataFrame(index=index)

    # For each prediction date, only events already available by that date are
    # eligible. This explicit loop favors correctness over premature optimization.
    for day in index:
        eligible = daily[daily["available_date"] <= day]
        if eligible.empty:
            continue

        eligible = eligible.copy()
        age_days = (day - eligible["event_date"]).dt.total_seconds() / 86400.0
        eligible = eligible[age_days >= 0].copy()
        if eligible.empty:
            continue
        eligible["age_days"] = age_days.loc[eligible.index]

        for window in windows:
            recent = eligible[eligible["age_days"] < window]
            prefix = f"events_{window}d"
            result.loc[day, f"{prefix}_count"] = float(len(recent))
            result.loc[day, f"{prefix}_conflict_count"] = float(recent["is_conflict"].sum())
            result.loc[day, f"{prefix}_intensity_sum"] = float(recent["intensity"].sum())
            result.loc[day, f"{prefix}_intensity_max"] = float(recent["intensity"].max()) if len(recent) else 0.0

        # Exponential decay makes a recent event matter more than an old one.
        decay = np.exp(-eligible["age_days"].to_numpy() / 5.0)
        result.loc[day, "event_pressure"] = float(np.sum(eligible["intensity"].to_numpy() * decay))
        result.loc[day, "conflict_pressure"] = float(
            np.sum(eligible["intensity"].to_numpy() * eligible["is_conflict"].to_numpy() * decay)
        )

    return result.fillna(0.0)


def merge_market_events(
    market: pd.DataFrame,
    event_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge event features onto market rows by calendar day."""
    if not isinstance(market.index, pd.DatetimeIndex):
        raise TypeError("market index must be a DatetimeIndex")
    left = market.copy()
    left.index = pd.to_datetime(left.index, utc=True).floor("D")
    right = event_features.copy()
    right.index = pd.to_datetime(right.index, utc=True).floor("D")
    return left.join(right, how="left").fillna(0.0)
