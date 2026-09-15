"""Financial evaluation of out-of-sample probability signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def backtest_long_only(predictions: pd.DataFrame, *, probability_column: str = "prob_up", close_column: str = "close", threshold: float = 0.5, transaction_cost_bps: float = 5.0, slippage_bps: float = 0.0, periods_per_year: int = 252) -> tuple[pd.DataFrame, dict[str, float]]:
    """Backtest a long/flat signal, charging explicit execution costs."""
    if not 0.0 <= threshold <= 1.0: raise ValueError("threshold must be between 0 and 1")
    if transaction_cost_bps < 0 or slippage_bps < 0: raise ValueError("costs cannot be negative")
    if periods_per_year <= 0: raise ValueError("periods_per_year must be positive")
    missing = {probability_column, close_column} - set(predictions.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = predictions[[probability_column, close_column]].copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    if frame.index.has_duplicates: raise ValueError("prediction index must not contain duplicate timestamps")
    frame = frame.sort_index()
    frame["prob_up"] = pd.to_numeric(frame[probability_column], errors="coerce")
    frame["close"] = pd.to_numeric(frame[close_column], errors="coerce")
    frame = frame.dropna(subset=["prob_up", "close"])
    if frame.empty: raise ValueError("No valid observations for financial backtest")
    if ((frame["prob_up"] < 0) | (frame["prob_up"] > 1)).any(): raise ValueError("probabilities must be between 0 and 1")
    if (frame["close"] <= 0).any(): raise ValueError("close prices must be positive")
    frame["signal"] = (frame["prob_up"] >= threshold).astype(float)
    frame["asset_return"] = frame["close"].shift(-1) / frame["close"] - 1.0
    frame["position_change"] = frame["signal"].diff().abs().fillna(frame["signal"].abs())
    frame["cost"] = frame["position_change"] * ((transaction_cost_bps + slippage_bps) / 10_000.0)
    frame["strategy_return"] = frame["signal"] * frame["asset_return"] - frame["cost"]
    frame = frame.dropna(subset=["asset_return"])
    frame["strategy_equity"] = (1.0 + frame["strategy_return"]).cumprod()
    frame["benchmark_equity"] = (1.0 + frame["asset_return"]).cumprod()
    r = frame["strategy_return"]
    years = len(frame) / periods_per_year
    std = r.std(ddof=1)
    downside_sq_mean = float(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))
    metrics = {
        "total_return": float(frame["strategy_equity"].iloc[-1] - 1),
        "cagr": float(frame["strategy_equity"].iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
        "max_drawdown": _max_drawdown(frame["strategy_equity"]),
        "annualized_volatility": float(std * np.sqrt(periods_per_year)) if len(frame) > 1 else np.nan,
        "sharpe": float(r.mean() / std * np.sqrt(periods_per_year)) if len(frame) > 1 and std > 0 else np.nan,
        "sortino": float(r.mean() / np.sqrt(downside_sq_mean) * np.sqrt(periods_per_year)) if len(frame) > 1 and downside_sq_mean > 0 else np.nan,
        "downside_deviation": float(np.sqrt(downside_sq_mean * periods_per_year)),
        "hit_rate": float((r > 0).mean()),
        "benchmark_total_return": float(frame["benchmark_equity"].iloc[-1] - 1),
        "benchmark_max_drawdown": _max_drawdown(frame["benchmark_equity"]),
        "average_position": float(frame["signal"].mean()),
        "turnover": float(frame["position_change"].sum()),
        "total_cost": float(frame["cost"].sum()),
        "final_equity": float(frame["strategy_equity"].iloc[-1]),
        "observations": float(len(frame)),
    }
    return frame, metrics
