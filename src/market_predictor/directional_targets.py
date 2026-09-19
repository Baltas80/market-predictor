"""Research-only directional targets with explicit downside labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_directional_target(
    df: pd.DataFrame,
    horizon: int = 5,
    neutral_threshold: float = 0.0,
) -> pd.Series:
    """Return -1 DOWN, 0 NEUTRAL, +1 UP for the future return.

    This research target is deliberately separate from the production binary
    target. The final ``horizon`` observations remain NaN because their future
    labels are unavailable at decision time.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if neutral_threshold < 0:
        raise ValueError("neutral_threshold must be >= 0")
    if "close" not in df.columns:
        raise ValueError("Missing required column: close")

    future_close = df["close"].shift(-horizon)
    future_return = future_close / df["close"] - 1.0
    target = pd.Series(np.nan, index=df.index, dtype="float64")
    target.loc[future_return > neutral_threshold] = 1.0
    target.loc[future_return < -neutral_threshold] = -1.0
    target.loc[future_return.abs() <= neutral_threshold] = 0.0
    return target.where(future_close.notna())
