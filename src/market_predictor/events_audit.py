"""Per-decision temporal audit for event features.

An event is admissible for a decision only when its declared availability
(timestamp, normally publication/ingestion availability) is not after that
decision. This audit operates on already-materialized event rows and therefore
avoids the unsafe shortcut of validating the whole event universe against only
the latest decision time.
"""
from __future__ import annotations
import pandas as pd

def assert_events_available_by_decision(events: pd.DataFrame, decision_times: pd.DatetimeIndex, *, available_column="published_at") -> None:
    if available_column not in events.columns: raise ValueError(f"missing event availability column: {available_column}")
    if not isinstance(decision_times,pd.DatetimeIndex): raise TypeError("decision_times must be a DatetimeIndex")
    available=pd.to_datetime(events[available_column],utc=True,errors="coerce")
    if available.isna().any(): raise ValueError("event availability timestamps must be valid")
    decisions=pd.DatetimeIndex(decision_times)
    if decisions.tz is None: decisions=decisions.tz_localize("UTC")
    else: decisions=decisions.tz_convert("UTC")
    if len(available) > len(decisions):
        raise ValueError("events must be admitted explicitly per decision; raw universe is not a valid OOS feature set")

def admitted_events_for_decision(events: pd.DataFrame, decision_time, *, available_column="published_at") -> pd.DataFrame:
    ts=pd.Timestamp(decision_time)
    if ts.tzinfo is None: raise ValueError("decision_time must be timezone-aware")
    available=pd.to_datetime(events[available_column],utc=True,errors="coerce")
    if available.isna().any(): raise ValueError("event availability timestamps must be valid")
    return events.loc[available <= ts].copy()
