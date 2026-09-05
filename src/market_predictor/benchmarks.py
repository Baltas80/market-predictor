"""Simple baselines for honest out-of-sample comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold_returns(close: pd.Series) -> pd.Series:
    """Return the one-step asset return used by the financial benchmark."""
    price = pd.to_numeric(close, errors="coerce")
    if (price.dropna() <= 0).any():
        raise ValueError("close prices must be positive")
    return price.pct_change().dropna()


def random_signal(probabilities: pd.Series, *, seed: int = 42) -> pd.Series:
    """Create a deterministic 0/1 baseline with the same observation index."""
    values = pd.to_numeric(probabilities, errors="coerce")
    if values.empty or not values.notna().all() or not values.between(0.0, 1.0).all():
        raise ValueError("probabilities must contain only finite numeric values in [0, 1]")
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.integers(0, 2, size=len(values)),
        index=values.index,
        name="random_signal",
    )
