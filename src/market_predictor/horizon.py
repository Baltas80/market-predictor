"""Explicit multi-horizon signal construction for robustness studies."""

from __future__ import annotations

import pandas as pd


def future_return_target(close: pd.Series, horizons: tuple[int, ...] = (1, 3, 5, 10, 20, 60)) -> pd.DataFrame:
    """Build future directional targets without filling unavailable labels."""
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError("horizons must contain positive integers")
    price = pd.to_numeric(close, errors="coerce")
    if (price.dropna() <= 0).any():
        raise ValueError("close prices must be positive")
    out = pd.DataFrame(index=close.index)
    for horizon in horizons:
        future = price.shift(-horizon)
        out[f"target_{horizon}d"] = (future > price).astype("float64").where(future.notna())
    return out
