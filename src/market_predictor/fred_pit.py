"""Audits for FRED vintage-aware point-in-time ingestion."""
from __future__ import annotations

import pandas as pd


def audit_fred_point_in_time(observations: pd.DataFrame) -> pd.DataFrame:
    """Return one audit row per FRED observation/vintage.

    FRED's realtime fields provide vintage dates, not a guaranteed intraday
    release timestamp. The audit therefore treats ``realtime_start`` as the
    earliest defensible date-level availability marker and never invents a
    release clock time.
    """
    required = {"series_id", "date", "value", "realtime_start", "realtime_end"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"FRED observations missing columns: {sorted(missing)}")
    data = observations.copy()
    for column in ("date", "realtime_start", "realtime_end"):
        data[column] = pd.to_datetime(data[column], utc=True, errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    if data[["date", "realtime_start", "realtime_end", "value"]].isna().any().any():
        raise ValueError("FRED PIT audit found invalid timestamps or values")
    if (data["realtime_end"] < data["realtime_start"]).any():
        raise ValueError("FRED realtime_end cannot precede realtime_start")
    if (data["realtime_start"].dt.normalize() < data["date"].dt.normalize()).any():
        raise ValueError("FRED vintage cannot become available before its observation date")
    duplicate_key = ["series_id", "date", "realtime_start"]
    if data.duplicated(duplicate_key).any():
        raise ValueError("duplicate FRED vintage keys")
    return data.sort_values(["series_id", "date", "realtime_start"]).reset_index(drop=True)


def assert_fred_point_in_time(observations: pd.DataFrame) -> None:
    """Fail closed if FRED vintage metadata is not internally consistent."""
    audit_fred_point_in_time(observations)


def fred_availability_cutoff(decision_time: str | pd.Timestamp, *, conservative_session_lag: int = 1) -> pd.Timestamp:
    """Return the date-level FRED cutoff used by the historical aligner."""
    if conservative_session_lag < 0:
        raise ValueError("conservative_session_lag must be >= 0")
    decision = pd.Timestamp(decision_time)
    if decision.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    return (decision.tz_convert("UTC").normalize() - pd.Timedelta(value=int(conservative_session_lag), unit="D")).tz_convert("UTC")
