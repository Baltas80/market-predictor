"""Temporal audit primitives for historical point-in-time data."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalAudit:
    checked: int
    violations: int
    weekend_rows: int
    non_monotonic: bool


def audit_availability(
    frame: pd.DataFrame,
    *,
    decision_column: str = "decision_time",
    available_column: str = "available_at",
) -> TemporalAudit:
    """Audit that every admitted record was available by its decision time."""
    for column in (decision_column, available_column):
        if column not in frame.columns:
            raise ValueError(f"missing temporal column: {column}")
    decision = pd.to_datetime(frame[decision_column], utc=True, errors="coerce")
    available = pd.to_datetime(frame[available_column], utc=True, errors="coerce")
    if decision.isna().any() or available.isna().any():
        raise ValueError("invalid timestamp in temporal audit")
    violations = int((available > decision).sum())
    weekend_rows = int((decision.dt.dayofweek >= 5).sum())
    non_monotonic = not decision.is_monotonic_increasing
    return TemporalAudit(len(frame), violations, weekend_rows, non_monotonic)


def assert_no_future_information(frame: pd.DataFrame, *, decision_column: str = "decision_time", available_column: str = "available_at") -> TemporalAudit:
    """Fail closed on any future information, invalid ordering, or weekend decision."""
    result = audit_availability(frame, decision_column=decision_column, available_column=available_column)
    if result.violations:
        raise ValueError(f"future information detected: {result.violations} rows")
    if result.non_monotonic:
        raise ValueError("decision timestamps must be chronological")
    if result.weekend_rows:
        raise ValueError(f"weekend decision timestamps detected: {result.weekend_rows} rows")
    return result
