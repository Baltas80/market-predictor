"""Point-in-time auditing for FRED vintage observations."""
from __future__ import annotations

import pandas as pd


def audit_fred_point_in_time(observations: pd.DataFrame) -> None:
    required = {"series_id", "date", "value", "realtime_start", "realtime_end"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"FRED observations missing columns: {sorted(missing)}")
    frame = observations.copy()
    for column in ("date", "realtime_start", "realtime_end"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if frame[["date", "realtime_start", "realtime_end"]].isna().any().any():
        raise ValueError("FRED vintage timestamps must be valid")
    if (frame["realtime_end"] < frame["realtime_start"]).any():
        raise ValueError("FRED realtime_end must not precede realtime_start")
    if (frame["realtime_start"] < frame["date"]).any():
        raise ValueError("FRED vintage cannot become available before the observation date")
    if not pd.to_numeric(frame["value"], errors="coerce").notna().all():
        raise ValueError("FRED values must be numeric")


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
