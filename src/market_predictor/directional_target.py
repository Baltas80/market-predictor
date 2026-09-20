"""Exhaustive directional target for experimental research.

The neutrality band is fixed ex ante for this experiment and is never learned
from OOS data. It is expressed as a simple future return threshold.
"""
from __future__ import annotations

import pandas as pd


DIRECTION_UP = 1
DIRECTION_FLAT = 0
DIRECTION_DOWN = -1
DIRECTION_CLASSES = (DIRECTION_DOWN, DIRECTION_FLAT, DIRECTION_UP)

# Frozen protocol parameter: +/-10 basis points around the current close.
DEFAULT_FLAT_RETURN_THRESHOLD = 0.001


def make_directional_target(
    df: pd.DataFrame,
    horizon: int = 5,
    flat_return_threshold: float = DEFAULT_FLAT_RETURN_THRESHOLD,
) -> pd.Series:
    """Return {-1, 0, +1} from the future simple return.

    DOWN: future/current - 1 < -threshold
    FLAT: abs(future/current - 1) <= threshold
    UP:   future/current - 1 > threshold

    The threshold is an ex-ante protocol constant and must not be tuned on OOS.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if "close" not in df.columns:
        raise ValueError("Missing required column: close")
    if flat_return_threshold < 0:
        raise ValueError("flat_return_threshold must be >= 0")

    future_close = df["close"].shift(-horizon)
    future_return = future_close / df["close"] - 1.0
    target = pd.Series(pd.NA, index=df.index, dtype="Int64")
    known = future_return.notna() & df["close"].notna() & (df["close"] != 0)
    target.loc[known & (future_return > flat_return_threshold)] = DIRECTION_UP
    target.loc[known & (future_return < -flat_return_threshold)] = DIRECTION_DOWN
    target.loc[
        known
        & (future_return >= -flat_return_threshold)
        & (future_return <= flat_return_threshold)
    ] = DIRECTION_FLAT
    return target
