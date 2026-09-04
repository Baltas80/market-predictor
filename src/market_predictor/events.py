"""Point-in-time geopolitical and macro event feature engineering."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_EVENT_COLUMNS = {"event_time", "available_time", "event_type", "intensity"}
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


def _normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    validate_events(events)
    out = events.copy()
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True)
    out["available_time"] = pd.to_datetime(out["available_time"], utc=True)
    out["intensity"] = pd.to_numeric(out["intensity"], errors="coerce").fillna(0.0)
    out["is_conflict"] = out["event_type"].astype(str).str.lower().isin(
        {"war", "armed_conflict", "military_attack", "terrorism", "civil_unrest", "sanction"}
    ).astype(int)
    return out


def build_event_features(
    events: pd.DataFrame,
    prediction_times: Iterable[pd.Timestamp],
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Build leakage-safe event features for exact prediction timestamps.

    An event is eligible only when ``available_time <= prediction_time`` and
    ``event_time <= prediction_time``. This is stricter than joining by date:
    an event published after a market close cannot leak into that day's signal.
    """
    if any(window < 1 for window in windows):
        raise ValueError("windows must contain positive day counts")

    events_norm = _normalize_events(events)
    index = pd.DatetimeIndex(pd.to_datetime(list(prediction_times), utc=True))
    index = index.sort_values().unique()
    result = pd.DataFrame(index=index)

    for timestamp in index:
        eligible = events_norm[
            (events_norm["available_time"] <= timestamp)
            & (events_norm["event_time"] <= timestamp)
        ].copy()
        if eligible.empty:
            continue

        age_days = (timestamp - eligible["event_time"]).dt.total_seconds() / 86400.0
        eligible["age_days"] = age_days

        for window in windows:
            recent = eligible[eligible["age_days"] < window]
            prefix = f"events_{window}d"
            result.loc[timestamp, f"{prefix}_count"] = float(len(recent))
            result.loc[timestamp, f"{prefix}_conflict_count"] = float(recent["is_conflict"].sum())
            result.loc[timestamp, f"{prefix}_intensity_sum"] = float(recent["intensity"].sum())
            result.loc[timestamp, f"{prefix}_intensity_max"] = (
                float(recent["intensity"].max()) if len(recent) else 0.0
            )

        decay = np.exp(-eligible["age_days"].to_numpy() / 5.0)
        result.loc[timestamp, "event_pressure"] = float(
            np.sum(eligible["intensity"].to_numpy() * decay)
        )
        result.loc[timestamp, "conflict_pressure"] = float(
            np.sum(eligible["intensity"].to_numpy() * eligible["is_conflict"].to_numpy() * decay)
        )

    return result.fillna(0.0)


def merge_market_events(market: pd.DataFrame, event_features: pd.DataFrame) -> pd.DataFrame:
    """Merge event features without changing the market observation timestamps."""
    if not isinstance(market.index, pd.DatetimeIndex):
        raise TypeError("market index must be a DatetimeIndex")
    left = market.copy()
    left.index = pd.to_datetime(left.index, utc=True)
    right = event_features.copy()
    right.index = pd.to_datetime(right.index, utc=True)
    return left.join(right, how="left").fillna(0.0)
