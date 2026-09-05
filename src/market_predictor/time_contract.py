"""Explicit temporal contract for point-in-time research observations."""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class ObservationWindow:
    """Information available for a decision at ``decision_time``.

    ``available_at`` is the timestamp at which the information may legally
    enter features. ``target_end`` is the end of the future outcome window.
    """

    available_at: pd.Timestamp
    decision_time: pd.Timestamp
    target_end: pd.Timestamp

    def validate(self) -> None:
        available = pd.Timestamp(self.available_at)
        decision = pd.Timestamp(self.decision_time)
        target_end = pd.Timestamp(self.target_end)
        if available.tzinfo is None or decision.tzinfo is None or target_end.tzinfo is None:
            raise ValueError("Temporal contract timestamps must be timezone-aware")
        if available > decision:
            raise ValueError("Information cannot be used before it is available")
        if target_end <= decision:
            raise ValueError("Target end must be strictly after decision time")


def assert_point_in_time(
    available_at: pd.Series,
    decision_time: pd.Series,
    target_end: pd.Series,
) -> None:
    """Validate aligned timestamp series under the temporal contract."""
    if not (len(available_at) == len(decision_time) == len(target_end)):
        raise ValueError("Temporal contract series must have identical lengths")
    available = pd.to_datetime(available_at, utc=True, errors="coerce")
    decision = pd.to_datetime(decision_time, utc=True, errors="coerce")
    target = pd.to_datetime(target_end, utc=True, errors="coerce")
    if available.isna().any() or decision.isna().any() or target.isna().any():
        raise ValueError("Temporal contract contains invalid timestamps")
    if (available > decision).any():
        raise ValueError("Point-in-time violation: information becomes available after decision")
    if (target <= decision).any():
        raise ValueError("Invalid target horizon: target ends at or before decision")


def assert_market_target_contract(index: pd.Index, horizon: int) -> None:
    """Validate chronological market timestamps against a future target horizon.

    The decision timestamp is the current market observation and the target end
    is the market observation ``horizon`` steps ahead. The final observations
    without a known target are excluded from the temporal check.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("market index must be a DatetimeIndex")

    market = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    if not market.is_monotonic_increasing:
        raise ValueError("Market index must be chronological")
    if market.has_duplicates:
        raise ValueError("Market index must not contain duplicates")

    decision = pd.Series(market, index=range(len(market)))
    target_end = decision.shift(-horizon)
    valid = target_end.notna()
    assert_point_in_time(decision.loc[valid], decision.loc[valid], target_end.loc[valid])


def next_session_decision_time(
    index: pd.DatetimeIndex,
    available_at: pd.Timestamp,
) -> pd.Timestamp:
    """Return the first market timestamp strictly after information availability."""
    market = pd.DatetimeIndex(index)
    if market.tz is None:
        market = market.tz_localize("UTC")
    else:
        market = market.tz_convert("UTC")
    available = pd.Timestamp(available_at)
    if available.tzinfo is None:
        available = available.tz_localize("UTC")
    else:
        available = available.tz_convert("UTC")
    future = market[market > available]
    if len(future) == 0:
        raise ValueError("No market decision timestamp exists after availability")
    return future[0]
