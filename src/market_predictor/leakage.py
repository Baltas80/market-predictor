"""Leakage and point-in-time integrity checks for research datasets.

These checks are intentionally strict: a failed check should stop a dataset
from reaching the final OOS experiment rather than silently dropping rows.
"""

from __future__ import annotations

import pandas as pd


def assert_monotonic_unique_index(index: pd.DatetimeIndex) -> None:
    """Require a chronological, unique decision-time index."""
    idx = pd.DatetimeIndex(index)
    if not idx.is_monotonic_increasing:
        raise ValueError("Decision-time index must be chronological")
    if idx.has_duplicates:
        raise ValueError("Decision-time index contains duplicate timestamps")


def assert_target_is_future(
    frame: pd.DataFrame,
    *,
    decision_column: str = "decision_time",
    target_end_column: str = "target_end",
) -> None:
    """Ensure every target is strictly after its decision timestamp."""
    for column in (decision_column, target_end_column):
        if column not in frame.columns:
            raise ValueError(f"Missing required leakage column: {column}")
    decision = pd.to_datetime(frame[decision_column], errors="coerce", utc=True)
    target_end = pd.to_datetime(frame[target_end_column], errors="coerce", utc=True)
    invalid = target_end.notna() & decision.notna() & (target_end <= decision)
    if invalid.any():
        raise ValueError("Target horizon contains observations at or before decision time")


def assert_event_availability_before_decision(
    events: pd.DataFrame,
    *,
    available_column: str = "published_at",
    decision_times: pd.DatetimeIndex,
) -> None:
    """Verify that every event admitted to a decision is actually available."""
    if available_column not in events.columns:
        raise ValueError(f"Missing event availability column: {available_column}")
    available = pd.to_datetime(events[available_column], errors="coerce", utc=True)
    decisions = pd.DatetimeIndex(decision_times)
    if decisions.tz is None:
        decisions = decisions.tz_localize("UTC")
    else:
        decisions = decisions.tz_convert("UTC")
    invalid = available.notna() & (available > decisions.max())
    if invalid.any():
        raise ValueError("Event set contains information unavailable by the latest decision time")


def audit_feature_columns(
    columns: list[str] | pd.Index,
    *,
    forbidden_tokens: tuple[str, ...] = ("target", "future", "forward", "lead"),
) -> list[str]:
    """Return suspicious feature names for human review.

    This is a heuristic audit, not proof of absence of leakage. It catches
    common naming mistakes while code-level PIT tests provide the stronger
    guarantee.
    """
    suspicious = []
    for column in columns:
        name = str(column).lower()
        if any(token in name for token in forbidden_tokens):
            suspicious.append(str(column))
    return suspicious
