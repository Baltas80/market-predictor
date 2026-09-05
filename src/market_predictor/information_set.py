"""Build auditable point-in-time information sets for historical decisions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .events_audit import admitted_events_for_decision


@dataclass(frozen=True)
class InformationSet:
    """Information admitted to one historical decision timestamp."""

    decision_time: pd.Timestamp
    market_row: pd.Series
    macro_rows: pd.DataFrame
    event_rows: pd.DataFrame


def build_information_set(
    *,
    decision_time: pd.Timestamp,
    market: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    market_available_at: str | None = None,
    macro_available_at: str = "vintage_start",
    event_available_at: str = "available_at",
) -> InformationSet:
    """Return only data demonstrably available by ``decision_time``.

    The market table is interpreted as session observations whose index is the
    decision clock. Macro data is admitted using its vintage availability;
    events use their explicit availability timestamp. No forward fill or
    future-value substitution is performed here.
    """
    decision = pd.Timestamp(decision_time)
    if decision.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    if not isinstance(market.index, pd.DatetimeIndex):
        raise TypeError("market must have a DatetimeIndex")
    if not market.index.is_monotonic_increasing or not market.index.is_unique:
        raise ValueError("market index must be chronological and unique")

    eligible_market = market.loc[market.index <= decision]
    if eligible_market.empty:
        raise ValueError("no market observation is available at decision_time")
    market_row = eligible_market.iloc[-1]

    if macro is None:
        macro_rows = pd.DataFrame()
    else:
        if macro_available_at not in macro.columns:
            raise ValueError(f"macro missing availability column: {macro_available_at}")
        available = pd.to_datetime(macro[macro_available_at], utc=True, errors="coerce")
        if available.isna().any():
            raise ValueError("macro availability contains invalid timestamps")
        macro_rows = macro.loc[available <= decision].copy()

    if events is None:
        event_rows = pd.DataFrame()
    else:
        event_rows = admitted_events_for_decision(
            events,
            decision,
            available_column=event_available_at,
        ).copy()

    return InformationSet(decision, market_row, macro_rows, event_rows)


def build_information_set_history(
    decision_times: pd.DatetimeIndex,
    *,
    market: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
) -> list[InformationSet]:
    """Reconstruct the information set independently for every decision."""
    if not isinstance(decision_times, pd.DatetimeIndex):
        raise TypeError("decision_times must be a DatetimeIndex")
    if not decision_times.is_monotonic_increasing or not decision_times.is_unique:
        raise ValueError("decision_times must be chronological and unique")
    return [
        build_information_set(
            decision_time=decision,
            market=market,
            macro=macro,
            events=events,
        )
        for decision in decision_times
    ]
