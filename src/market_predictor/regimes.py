"""Causal market-regime labels for out-of-sample diagnostics."""

from __future__ import annotations

import pandas as pd


def causal_regimes(
    close: pd.Series,
    *,
    volatility_window: int = 20,
    threshold_window: int = 252,
    high_vol_quantile: float = 0.80,
    crisis_return_window: int = 20,
    crisis_return_threshold: float = -0.10,
) -> pd.Series:
    """Label normal/high-vol/crisis using information available at each timestamp only.

    Thresholds are computed from strictly prior observations via ``shift(1)``.
    """
    if volatility_window < 2 or threshold_window < 2 or crisis_return_window < 1:
        raise ValueError("windows must be positive and volatility_window >= 2")
    if not 0.0 < high_vol_quantile < 1.0:
        raise ValueError("high_vol_quantile must be in (0, 1)")
    price = pd.to_numeric(close, errors="coerce")
    if price.empty or not price.notna().all() or (price <= 0).any():
        raise ValueError("close must contain only positive numeric values")
    returns = price.pct_change()
    vol = returns.rolling(volatility_window, min_periods=volatility_window).std()
    past_vol = vol.shift(1)
    vol_threshold = past_vol.rolling(threshold_window, min_periods=threshold_window).quantile(high_vol_quantile)
    past_crisis_return = price.pct_change(crisis_return_window).shift(1)
    labels = pd.Series("normal", index=price.index, name="regime", dtype="string")
    labels.loc[vol > vol_threshold] = "high_volatility"
    labels.loc[past_crisis_return <= crisis_return_threshold] = "crisis"
    return labels
