"""Convert timestamped market events into leakage-safe time-series features."""

from __future__ import annotations

import pandas as pd

from .event_schema import EventCategory, MarketEvent


def events_to_features(
    index: pd.DatetimeIndex,
    events: list[MarketEvent],
    *,
    half_life_days: float = 7.0,
) -> pd.DataFrame:
    """Aggregate only information available at each market timestamp.

    ``available_at`` gates admission. ``published_at`` and ``event_time`` are
    retained as descriptive provenance and are never substituted for it.
    """
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")

    original_idx = pd.DatetimeIndex(index)
    calc_idx = original_idx.tz_localize("UTC") if original_idx.tz is None else original_idx.tz_convert("UTC")
    columns = [f"event_{category.value}" for category in EventCategory]
    out = pd.DataFrame(0.0, index=original_idx, columns=columns)
    out["event_total_pressure"] = 0.0
    out["event_count"] = 0.0
    out["event_surprise"] = 0.0

    for event in events:
        available = pd.Timestamp(event.available_at if event.available_at is not None else event.published_at)
        available = available.tz_localize("UTC") if available.tzinfo is None else available.tz_convert("UTC")
        elapsed = (calc_idx - available).total_seconds() / 86400.0
        known = elapsed >= 0
        weight = pd.Series(0.0, index=original_idx)
        weight.loc[known] = 2.0 ** (-elapsed[known] / half_life_days)
        pressure = weight * float(event.severity)
        column = f"event_{event.category.value}"
        out[column] += pressure
        out["event_total_pressure"] += pressure
        out["event_count"] += weight
        out["event_surprise"] += pressure * float(event.surprise)

    return out
