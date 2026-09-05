"""Isolated AI overlay adapter for future experiment D.

This module deliberately does not train, tune, or mutate experiment C. It only
admits timestamped AI signals when they are available to the corresponding C
decision clock and preserves the C OOS index.
"""
from __future__ import annotations

import pandas as pd

from .experiment_d import DExperimentSpec


def prepare_ai_overlay(c_predictions: pd.DataFrame, ai_signals: pd.DataFrame, *, available_column: str = "available_at") -> pd.DataFrame:
    """Return AI signals with their first eligible C decision timestamp.

    A signal is not admitted merely because it is before the last OOS decision.
    Its availability is mapped independently to the first C decision at or after
    the availability timestamp, preventing a future signal from being reused at
    an earlier decision.
    """
    DExperimentSpec().validate()
    if not c_predictions.index.is_monotonic_increasing or c_predictions.index.has_duplicates:
        raise ValueError("C predictions must have a unique chronological OOS index")
    if not isinstance(c_predictions.index, pd.DatetimeIndex) or c_predictions.index.tz is None:
        raise ValueError("C prediction index must be a timezone-aware DatetimeIndex")
    if available_column not in ai_signals.columns:
        raise ValueError(f"AI signals require {available_column}")
    signals = ai_signals.copy()
    signals[available_column] = pd.to_datetime(signals[available_column], utc=True, errors="coerce")
    if signals[available_column].isna().any():
        raise ValueError("AI availability timestamps must be valid")
    decisions = pd.DatetimeIndex(c_predictions.index).tz_convert("UTC")
    available = signals[available_column].dt.tz_convert("UTC")
    positions = decisions.searchsorted(available, side="left")
    in_lockbox = positions < len(decisions)
    admitted = signals.loc[in_lockbox].copy()
    admitted["first_eligible_decision"] = decisions.take(positions[in_lockbox])
    return admitted.sort_values("first_eligible_decision").reset_index(drop=True)


def assert_d_uses_c_lockbox(c_predictions: pd.DataFrame, d_predictions: pd.DataFrame) -> None:
    """Fail closed if D is evaluated on anything other than C's OOS observations."""
    if not c_predictions.index.equals(d_predictions.index):
        raise ValueError("D must use exactly C's OOS prediction index")
