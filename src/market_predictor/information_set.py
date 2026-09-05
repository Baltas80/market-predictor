"""Build auditable point-in-time information sets for historical decisions."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .events_audit import admitted_events_for_decision


@dataclass(frozen=True)
class InformationSet:
    """Information admitted to one historical decision timestamp."""

    decision_time: pd.Timestamp
    market_row: pd.Series
    macro_rows: pd.DataFrame
    event_rows: pd.DataFrame


def _require_aware_index(index: pd.DatetimeIndex, name: str) -> None:
    if index.tz is None:
        raise ValueError(f"{name} index must be timezone-aware")


def _require_compatible_timezone(index: pd.DatetimeIndex, decision: pd.Timestamp, name: str) -> None:
    _require_aware_index(index, name)
    # Different aware timezone representations are valid when converted to the
    # same UTC clock. The important invariant is that naive timestamps can never
    # be silently compared with an aware decision clock.
    try:
        index.tz_convert("UTC")
        decision.tz_convert("UTC")
    except Exception as exc:
        raise ValueError(f"{name} timezone cannot be normalized to UTC") from exc


def build_information_set(*, decision_time: pd.Timestamp, market: pd.DataFrame, macro: pd.DataFrame | None = None, events: pd.DataFrame | None = None, market_available_at: str | None = None, macro_available_at: str = "vintage_start", event_available_at: str = "available_at") -> InformationSet:
    """Return only data demonstrably available by ``decision_time``."""
    decision = pd.Timestamp(decision_time)
    if decision.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    if not isinstance(market.index, pd.DatetimeIndex):
        raise TypeError("market must have a DatetimeIndex")
    if not market.index.is_monotonic_increasing or not market.index.is_unique:
        raise ValueError("market index must be chronological and unique")
    _require_compatible_timezone(market.index, decision, "market")

    if market_available_at is not None:
        if market_available_at not in market.columns:
            raise ValueError(f"market missing availability column: {market_available_at}")
        market_available = pd.to_datetime(market[market_available_at], utc=True, errors="coerce")
        if market_available.isna().any():
            raise ValueError("market availability contains invalid timestamps")
        eligible_market = market.loc[market_available <= decision]
    else:
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
        event_rows = admitted_events_for_decision(events, decision, available_column=event_available_at).copy()
    return InformationSet(decision, market_row, macro_rows, event_rows)


def build_information_set_history(decision_times: pd.DatetimeIndex, *, market: pd.DataFrame, macro: pd.DataFrame | None = None, events: pd.DataFrame | None = None) -> list[InformationSet]:
    """Reconstruct the information set independently for every decision."""
    if not isinstance(decision_times, pd.DatetimeIndex):
        raise TypeError("decision_times must be a DatetimeIndex")
    if not decision_times.is_monotonic_increasing or not decision_times.is_unique:
        raise ValueError("decision_times must be chronological and unique")
    if decision_times.tz is None:
        raise ValueError("decision_times index must be timezone-aware")
    return [build_information_set(decision_time=decision, market=market, macro=macro, events=events) for decision in decision_times]
