"""Causal market-regime labels for post-hoc robustness analysis."""

from __future__ import annotations

import pandas as pd


def volatility_regime(close: pd.Series, window: int = 20, quantile: float = 0.8) -> pd.Series:
    """Label observations using only volatility information available at t."""
    if window < 2 or not 0.5 < quantile < 1.0:
        raise ValueError("invalid volatility regime parameters")
    returns = pd.to_numeric(close, errors="coerce").pct_change()
    vol = returns.rolling(window).std()
    expanding_threshold = vol.expanding(min_periods=window).quantile(quantile)
    return pd.Series(
        pd.NA,
        index=close.index,
        dtype="string",
    ).where(vol.isna(), vol.ge(expanding_threshold).map({True: "high_volatility", False: "normal"}))
