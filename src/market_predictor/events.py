"""Point-in-time geopolitical and macro event feature engineering."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_EVENT_COLUMNS = {"event_time", "available_time", "event_type", "intensity"}
DEFAULT_WINDOWS = (1, 3, 5, 10, 20)


def validate_events(events: pd.DataFrame) -> None:
    """Validate the minimum point-in-time event schema."""
    missing = REQUIRED_EVENT_COLUMNS - set(events.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")

    event_time = pd.to_datetime(events["event_time"], utc=True, errors="coerce")
    available_time = pd.to_datetime(events["available_time"], utc=True, errors="coerce")
    if event_time.isna().any() or available_time.isna().any():
        raise ValueError("event_time and available_time must contain valid timestamps")
    if (available_time < event_time).any():
        raise ValueError("available_time cannot precede event_time")


def _normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    validate_events(events)
    out = events.copy()
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True)
    out["available_time"] = pd.to_datetime(out["available_time"], utc=True)
    out["intensity"] = pd.to_numeric(out["intensity"], errors="coerce").fillna(0.0)
    out["is_conflict"] = out["event_type"].astype(str).str.lower().isin(
        {"war", "armed_conflict", "military_attack", "terrorism", "civil_unrest", "sanction", "material_conflict"}
    ).astype(float)
    return out.sort_values(["event_time", "available_time"]).reset_index(drop=True)


def _range_add(target: np.ndarray, start: int, stop: int, value: float) -> None:
    """Add a constant value to the half-open interval [start, stop)."""
    if start < stop:
        target[start] += value
        target[stop] -= value


def build_event_features(
    events: pd.DataFrame,
    prediction_times: Iterable[pd.Timestamp],
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Build scalable, leakage-safe event features for exact prediction times.

    An event enters the signal only at ``max(event_time, available_time)`` and
    expires according to each requested calendar-day window. This avoids the
    previous O(predictions × events) implementation and preserves timestamp-
    level protection against post-close publication leakage.
    """
    if not windows or any(window < 1 for window in windows):
        raise ValueError("windows must contain positive day counts")

    events_norm = _normalize_events(events)
    index = pd.DatetimeIndex(pd.to_datetime(list(prediction_times), utc=True)).sort_values().unique()
    result = pd.DataFrame(index=index)
    if len(index) == 0:
        return result
    if events_norm.empty:
        columns = []
        for window in windows:
            prefix = f"events_{window}d"
            columns += [f"{prefix}_count", f"{prefix}_conflict_count", f"{prefix}_intensity_sum", f"{prefix}_intensity_max"]
        columns += ["event_pressure", "conflict_pressure"]
        return pd.DataFrame(0.0, index=index, columns=columns)

    prediction_ns = index.view("i8")
    available_ns = events_norm["available_time"].array.asi8
    event_ns = events_norm["event_time"].array.asi8
    start_ns = np.maximum(available_ns, event_ns)
    starts = np.searchsorted(prediction_ns, start_ns, side="left")

    for window in windows:
        prefix = f"events_{window}d"
        count_diff = np.zeros(len(index) + 1, dtype=float)
        conflict_diff = np.zeros(len(index) + 1, dtype=float)
        intensity_diff = np.zeros(len(index) + 1, dtype=float)
        intensity_max = np.zeros(len(index), dtype=float)

        end_ns = event_ns + int(window * 86_400_000_000_000)
        ends = np.searchsorted(prediction_ns, end_ns, side="left")
        for row, (start, end) in enumerate(zip(starts, ends)):
            if start >= end or start >= len(index):
                continue
            end = min(end, len(index))
            intensity = float(events_norm.iloc[row]["intensity"])
            conflict = float(events_norm.iloc[row]["is_conflict"])
            _range_add(count_diff, start, end, 1.0)
            _range_add(conflict_diff, start, end, conflict)
            _range_add(intensity_diff, start, end, intensity)
            intensity_max[start:end] = np.maximum(intensity_max[start:end], intensity)

        result[f"{prefix}_count"] = np.cumsum(count_diff[:-1])
        result[f"{prefix}_conflict_count"] = np.cumsum(conflict_diff[:-1])
        result[f"{prefix}_intensity_sum"] = np.cumsum(intensity_diff[:-1])
        result[f"{prefix}_intensity_max"] = intensity_max

    # Exponentially decayed pressure. A difference array stores the coefficient
    # over each active interval, then the prediction-time exponential is applied.
    pressure_coeff = np.zeros(len(index) + 1, dtype=float)
    conflict_coeff = np.zeros(len(index) + 1, dtype=float)
    decay_seconds = 5.0 * 86_400.0
    prediction_seconds = prediction_ns.astype(float) / 1_000_000_000.0
    for row, start in enumerate(starts):
        if start >= len(index):
            continue
        end = np.searchsorted(prediction_ns, event_ns[row] + int(20 * 86_400_000_000_000), side="left")
        end = min(end, len(index))
        if start >= end:
            continue
        event_seconds = event_ns[row] / 1_000_000_000.0
        intensity = float(events_norm.iloc[row]["intensity"])
        conflict = float(events_norm.iloc[row]["is_conflict"])
        coefficient = intensity * np.exp(event_seconds / decay_seconds)
        _range_add(pressure_coeff, start, end, coefficient)
        _range_add(conflict_coeff, start, end, coefficient * conflict)

    active_pressure = np.cumsum(pressure_coeff[:-1])
    active_conflict = np.cumsum(conflict_coeff[:-1])
    result["event_pressure"] = active_pressure * np.exp(-prediction_seconds / decay_seconds)
    result["conflict_pressure"] = active_conflict * np.exp(-prediction_seconds / decay_seconds)
    return result.fillna(0.0)


def merge_market_events(market: pd.DataFrame, event_features: pd.DataFrame) -> pd.DataFrame:
    """Merge event features without changing the market observation timestamps."""
    if not isinstance(market.index, pd.DatetimeIndex):
        raise TypeError("market index must be a DatetimeIndex")
    left = market.copy()
    left.index = pd.to_datetime(left.index, utc=True)
    right = event_features.copy()
    right.index = pd.to_datetime(right.index, utc=True)
    return left.join(right, how="left").fillna(0.0)
