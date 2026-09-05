"""Isolated AI overlay adapter for future experiment D.

This module deliberately does not train, tune, or mutate experiment C. It only
validates and joins timestamped AI signals to C on an identical OOS index.
"""
from __future__ import annotations

import pandas as pd

from .experiment_d import DExperimentSpec


def prepare_ai_overlay(c_predictions: pd.DataFrame, ai_signals: pd.DataFrame, *, available_column: str = "available_at") -> pd.DataFrame:
    """Join AI signals only when they were available by the C decision time."""
    DExperimentSpec().validate()
    if not c_predictions.index.is_monotonic_increasing or c_predictions.index.has_duplicates:
        raise ValueError("C predictions must have a unique chronological OOS index")
    if available_column not in ai_signals.columns:
        raise ValueError(f"AI signals require {available_column}")
    signals = ai_signals.copy()
    signals[available_column] = pd.to_datetime(signals[available_column], utc=True, errors="coerce")
    if signals[available_column].isna().any():
        raise ValueError("AI availability timestamps must be valid")
    decision = pd.DatetimeIndex(c_predictions.index)
    if decision.tz is None:
        raise ValueError("C prediction index must be timezone-aware")
    admitted = signals.loc[signals[available_column] <= decision.max()].copy()
    # AI remains an overlay table; no C prediction column is overwritten.
    return admitted


def assert_d_uses_c_lockbox(c_predictions: pd.DataFrame, d_predictions: pd.DataFrame) -> None:
    """Fail closed if D is evaluated on anything other than C's observations."""
    if not c_predictions.index.equals(d_predictions.index):
        raise ValueError("D must use exactly C's OOS prediction index")
