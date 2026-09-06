"""Audits for FRED vintage-aware point-in-time ingestion."""
from __future__ import annotations

import pandas as pd


def _parse_fred_timestamp(series: pd.Series) -> pd.Series:
    """Parse FRED timestamps while preserving its open-ended 9999-12-31 sentinel."""
    values = series.astype("string")
    open_ended = values.eq("9999-12-31")
    parsed = pd.to_datetime(values.mask(open_ended), utc=True, errors="coerce")
    if open_ended.any():
        parsed.loc[open_ended] = pd.Timestamp.max.tz_localize("UTC")
    return parsed


def audit_fred_point_in_time(observations: pd.DataFrame) -> pd.DataFrame:
    """Return one audit row per FRED observation/vintage.

    FRED real-time periods describe when a particular revision is the latest
    information available; they are not required to start on or after the
    observation date. In particular, the real-time start may precede the date
    being measured. The PIT audit therefore validates the temporal consistency
    of the real-time interval itself, but does not impose the invalid invariant
    ``realtime_start >= observation_date``.
    """
    required = {"series_id", "date", "value", "realtime_start", "realtime_end"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"FRED observations missing columns: {sorted(missing)}")
    data = observations.copy()
    data["date"] = pd.to_datetime(data["date"], utc=True, errors="coerce")
    data["realtime_start"] = pd.to_datetime(data["realtime_start"], utc=True, errors="coerce")
    data["realtime_end"] = _parse_fred_timestamp(data["realtime_end"])
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    if data[["date", "realtime_start", "realtime_end", "value"]].isna().any().any():
        raise ValueError("FRED PIT audit found invalid timestamps or values")
    if (data["realtime_end"] < data["realtime_start"]).any():
        raise ValueError("FRED realtime_end cannot precede realtime_start")
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
