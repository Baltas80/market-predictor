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
    """Aggregate events known at each market timestamp.

    Each row only receives events whose ``published_at`` is at or before that
    timestamp. Event influence decays exponentially after publication. This
    makes the transformation suitable for chronological backtesting.
    """
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out = pd.DataFrame(0.0, index=idx, columns=[
        f"event_{category.value}" for category in EventCategory
    ])
    out["event_total_pressure"] = 0.0
    out["event_count"] = 0.0
    out["event_surprise"] = 0.0

    for event in events:
        published = pd.Timestamp(event.published_at)
        if published.tzinfo is None:
            published = published.tz_localize("UTC")
        else:
            published = published.tz_convert("UTC")
        elapsed = (idx - published).total_seconds() / 86400.0
        known = elapsed >= 0
        weight = pd.Series(0.0, index=idx)
        weight.loc[known] = 2.0 ** (-elapsed[known] / half_life_days)
        pressure = weight * float(event.severity)
        column = f"event_{event.category.value}"
        out[column] += pressure
        out["event_total_pressure"] += pressure
        out["event_count"] += weight
        out["event_surprise"] += pressure * float(event.surprise)

    return out
