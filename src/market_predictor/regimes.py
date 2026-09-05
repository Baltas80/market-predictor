"""Causal market-regime labels for out-of-sample diagnostics."""

from __future__ import annotations
import pandas as pd

def causal_regimes(close: pd.Series, *, volatility_window=20, threshold_window=252, high_vol_quantile=.80, crisis_return_window=20, crisis_return_threshold=-.10) -> pd.Series:
    """Label regimes using only information available strictly before each timestamp."""
    if volatility_window < 2 or threshold_window < 2 or crisis_return_window < 1: raise ValueError("invalid windows")
    if not 0 < high_vol_quantile < 1: raise ValueError("high_vol_quantile must be in (0, 1)")
    price=pd.to_numeric(close,errors="coerce")
    if price.empty or not price.notna().all() or (price<=0).any(): raise ValueError("close must contain only positive numeric values")
    returns=price.pct_change(); vol=returns.rolling(volatility_window,min_periods=volatility_window).std()
    vol_threshold=vol.shift(1).rolling(threshold_window,min_periods=threshold_window).quantile(high_vol_quantile)
    past_crisis=price.pct_change(crisis_return_window).shift(1)
    labels=pd.Series("normal",index=price.index,name="regime",dtype="string")
    labels.loc[vol>vol_threshold]="high_volatility"; labels.loc[past_crisis<=crisis_return_threshold]="crisis"
    return labels
