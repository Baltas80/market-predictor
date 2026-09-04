"""Financial evaluation of out-of-sample probability signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def backtest_long_only(
    predictions: pd.DataFrame,
    *,
    probability_column: str = "prob_up",
    close_column: str = "close",
    threshold: float = 0.5,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 0.0,
    periods_per_year: int = 252,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Backtest a long/flat probability signal without execution leakage.

    The signal observed at time *t* is executed for the return from *t* to
    *t+1*. Costs are charged when the position changes. This function expects
    predictions and prices aligned on the same chronological index.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if transaction_cost_bps < 0 or slippage_bps < 0:
        raise ValueError("costs cannot be negative")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    required = {probability_column, close_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    frame = predictions[[probability_column, close_column]].copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()
    frame["prob_up"] = pd.to_numeric(frame[probability_column], errors="coerce")
    frame["close"] = pd.to_numeric(frame[close_column], errors="coerce")
    frame = frame.dropna(subset=["prob_up", "close"])
    if frame.empty:
        raise ValueError("No valid observations for financial backtest")
    if (frame["close"] <= 0).any():
        raise ValueError("close prices must be positive")

    frame["signal"] = (frame["prob_up"] >= threshold).astype(float)
    frame["asset_return"] = frame["close"].shift(-1) / frame["close"] - 1.0
    frame["position_change"] = frame["signal"].diff().abs().fillna(frame["signal"].abs())
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000.0
    frame["cost"] = frame["position_change"] * cost_rate
    frame["strategy_return"] = frame["signal"] * frame["asset_return"] - frame["cost"]
    frame = frame.dropna(subset=["asset_return"])
    frame["strategy_equity"] = (1.0 + frame["strategy_return"]).cumprod()
    frame["benchmark_equity"] = (1.0 + frame["asset_return"]).cumprod()

    strategy_returns = frame["strategy_return"]
    benchmark_returns = frame["asset_return"]
    years = len(frame) / periods_per_year
    cagr = float(frame["strategy_equity"].iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    volatility = float(strategy_returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(frame) > 1 else np.nan
    downside = strategy_returns.where(strategy_returns < 0, 0.0)
    downside_dev = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(frame) > 1 else np.nan
    sharpe = float(strategy_returns.mean() / strategy_returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(frame) > 1 and strategy_returns.std(ddof=1) > 0 else np.nan
    sortino = float(strategy_returns.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(frame) > 1 and downside.std(ddof=1) > 0 else np.nan
    metrics = {
        "total_return": float(frame["strategy_equity"].iloc[-1] - 1.0),
        "cagr": cagr,
        "max_drawdown": _max_drawdown(frame["strategy_equity"]),
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "hit_rate": float((strategy_returns > 0).mean()),
        "benchmark_total_return": float(frame["benchmark_equity"].iloc[-1] - 1.0),
        "benchmark_max_drawdown": _max_drawdown(frame["benchmark_equity"]),
        "average_position": float(frame["signal"].mean()),
        "turnover": float(frame["position_change"].sum()),
        "observations": float(len(frame)),
    }
    return frame, metrics
