"""Per-decision temporal audit for event features."""
from __future__ import annotations
import pandas as pd

def assert_events_available_by_decision(events: pd.DataFrame, *, decision_column="decision_time", available_column="published_at") -> None:
    """Reject any event row whose availability is after its decision timestamp."""
    missing={decision_column,available_column}-set(events.columns)
    if missing: raise ValueError(f"missing event audit columns: {sorted(missing)}")
    available=pd.to_datetime(events[available_column],utc=True,errors="coerce")
    decisions=pd.to_datetime(events[decision_column],utc=True,errors="coerce")
    if available.isna().any() or decisions.isna().any(): raise ValueError("event audit timestamps must be valid")
    invalid=available>decisions
    if invalid.any(): raise ValueError(f"{int(invalid.sum())} event rows are unavailable at decision time")

def admitted_events_for_decision(events: pd.DataFrame, decision_time, *, available_column="published_at") -> pd.DataFrame:
    """Return only events available at one timezone-aware decision time."""
    if available_column not in events.columns: raise ValueError(f"missing event availability column: {available_column}")
    ts=pd.Timestamp(decision_time)
    if ts.tzinfo is None: raise ValueError("decision_time must be timezone-aware")
    available=pd.to_datetime(events[available_column],utc=True,errors="coerce")
    if available.isna().any(): raise ValueError("event availability timestamps must be valid")
    return events.loc[available <= ts].copy()
