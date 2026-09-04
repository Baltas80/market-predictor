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

    The signal observed at time *t* is held for the return from *t* to *t+1*.
    Therefore the model cannot use the next close to determine its position.
    Costs are charged when the position changes, including the initial entry.
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
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("prediction index must not contain duplicate timestamps")
    frame["prob_up"] = pd.to_numeric(frame[probability_column], errors="coerce")
    frame["close"] = pd.to_numeric(frame[close_column], errors="coerce")
    frame = frame.dropna(subset=["prob_up", "close"])
    if frame.empty:
        raise ValueError("No valid observations for financial backtest")
    if ((frame["prob_up"] < 0) | (frame["prob_up"] > 1)).any():
        raise ValueError("probabilities must be between 0 and 1")
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
    years = len(frame) / periods_per_year
    std = strategy_returns.std(ddof=1)
    downside_sq_mean = float(np.mean(np.minimum(strategy_returns.to_numpy(), 0.0) ** 2))
    downside_dev = np.sqrt(downside_sq_mean * periods_per_year)
    cagr = float(frame["strategy_equity"].iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    volatility = float(std * np.sqrt(periods_per_year)) if len(frame) > 1 else np.nan
    sharpe = float(strategy_returns.mean() / std * np.sqrt(periods_per_year)) if len(frame) > 1 and std > 0 else np.nan
    sortino = float(strategy_returns.mean() / np.sqrt(downside_sq_mean) * np.sqrt(periods_per_year)) if len(frame) > 1 and downside_sq_mean > 0 else np.nan
    metrics = {
        "total_return": float(frame["strategy_equity"].iloc[-1] - 1.0),
        "cagr": cagr,
        "max_drawdown": _max_drawdown(frame["strategy_equity"]),
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "downside_deviation": float(downside_dev),
        "hit_rate": float((strategy_returns > 0).mean()),
        "benchmark_total_return": float(frame["benchmark_equity"].iloc[-1] - 1.0),
        "benchmark_max_drawdown": _max_drawdown(frame["benchmark_equity"]),
        "average_position": float(frame["signal"].mean()),
        "turnover": float(frame["position_change"].sum()),
        "observations": float(len(frame)),
    }
    return frame, metrics
