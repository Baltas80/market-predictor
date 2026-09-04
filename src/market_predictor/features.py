"""Feature engineering for OHLCV market data."""

from __future__ import annotations

import pandas as pd


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe features to an OHLCV dataframe.

    Expected columns: open, high, low, close, volume.
    Features use only current or historical observations.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy().sort_index()
    out["return_1d"] = out["close"].pct_change()
    out["return_5d"] = out["close"].pct_change(5)
    out["volatility_20d"] = out["return_1d"].rolling(20).std()
    out["sma_20"] = out["close"].rolling(20).mean()
    out["sma_50"] = out["close"].rolling(50).mean()
    out["price_to_sma20"] = out["close"] / out["sma_20"] - 1.0
    out["volume_change"] = out["volume"].pct_change()
    out["range_pct"] = (out["high"] - out["low"]) / out["close"]
    return out


def make_target(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Binary target: 1 when future close is above current close."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return (df["close"].shift(-horizon) > df["close"]).astype("float")
