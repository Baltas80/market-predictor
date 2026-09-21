"""Exhaustive directional target for experimental research.

This module does not alter the existing binary lockbox target. It defines an
independent three-state target from the future close:
- +1: future close is higher
- -1: future close is lower
- 0: future close is exactly unchanged

The target uses only the future observation as a label and never enters the
feature set.
"""
from __future__ import annotations

import pandas as pd


DIRECTION_UP = 1
DIRECTION_FLAT = 0
DIRECTION_DOWN = -1
DIRECTION_CLASSES = (DIRECTION_DOWN, DIRECTION_FLAT, DIRECTION_UP)


def make_directional_target(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Return an exhaustive {-1, 0, +1} future-direction target."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if "close" not in df.columns:
        raise ValueError("Missing required column: close")

    future_close = df["close"].shift(-horizon)
    target = pd.Series(pd.NA, index=df.index, dtype="Int64")
    known = future_close.notna()
    target.loc[known & (future_close > df["close"])] = DIRECTION_UP
    target.loc[known & (future_close < df["close"])] = DIRECTION_DOWN
    target.loc[known & (future_close == df["close"])] = DIRECTION_FLAT
    return target
