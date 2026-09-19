"""Directional LONG/SHORT/ABSTAIN evaluation for out-of-sample predictions.

Experimental research layer. It does not alter the binary training target or model.
The existing P(up) output is mapped to three actions using fixed thresholds:
- LONG when P(up) >= upper threshold
- SHORT when P(up) <= lower threshold
- ABSTAIN otherwise.

The signal at time t only affects the return from t to t+1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def directional_position(
    probability_up: pd.Series,
    *,
    upper_threshold: float = 0.55,
    lower_threshold: float = 0.45,
) -> pd.Series:
    """Map P(up) to LONG=1, ABSTAIN=0, SHORT=-1 without fitting thresholds."""
    if not 0.5 < upper_threshold <= 1.0:
        raise ValueError("upper_threshold must be > 0.5 and <= 1")
    if not 0.0 <= lower_threshold < 0.5:
        raise ValueError("lower_threshold must be < 0.5 and >= 0")
    if lower_threshold >= upper_threshold:
        raise ValueError("lower_threshold must be below upper_threshold")
    values = pd.to_numeric(probability_up, errors="coerce")
    if values.isna().any():
        raise ValueError("probabilities must be finite")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("probabilities must be between 0 and 1")
    return pd.Series(
        np.select(
            [values >= upper_threshold, values <= lower_threshold],
            [1, -1],
            default=0,
        ),
        index=probability_up.index,
        name="position",
        dtype=int,
    )


def backtest_directional(
    predictions: pd.DataFrame,
    *,
    probability_column: str = "prob_up",
    close_column: str = "close",
    upper_threshold: float = 0.55,
    lower_threshold: float = 0.45,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 0.0,
    periods_per_year: int = 252,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Backtest LONG/SHORT/ABSTAIN without execution leakage."""
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
        raise ValueError("No valid observations for directional backtest")
    if ((frame["prob_up"] < 0) | (frame["prob_up"] > 1)).any():
        raise ValueError("probabilities must be between 0 and 1")
    if (frame["close"] <= 0).any():
        raise ValueError("close prices must be positive")

    frame["position"] = directional_position(
        frame["prob_up"],
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
    )
    frame["asset_return"] = frame["close"].shift(-1) / frame["close"] - 1.0
    frame["position_change"] = frame["position"].diff().abs().fillna(frame["position"].abs())
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000.0
    frame["cost"] = frame["position_change"] * cost_rate
    frame["strategy_return"] = frame["position"] * frame["asset_return"] - frame["cost"]
    frame["long_strategy_return"] = (frame["position"] == 1).astype(float) * frame["asset_return"]
    frame["short_strategy_return"] = (frame["position"] == -1).astype(float) * (-frame["asset_return"])
    frame = frame.dropna(subset=["asset_return"])

    frame["strategy_equity"] = (1.0 + frame["strategy_return"]).cumprod()
    frame["benchmark_equity"] = (1.0 + frame["asset_return"]).cumprod()

    active = frame["position"] != 0
    long_active = frame["position"] == 1
    short_active = frame["position"] == -1
    realized_direction_correct = (
        ((frame["position"] == 1) & (frame["asset_return"] > 0))
        | ((frame["position"] == -1) & (frame["asset_return"] < 0))
    )
    long_count = int(long_active.sum())
    short_count = int(short_active.sum())
    active_count = int(active.sum())

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
        "benchmark_total_return": float(frame["benchmark_equity"].iloc[-1] - 1.0),
        "benchmark_max_drawdown": _max_drawdown(frame["benchmark_equity"]),
        "average_position": float(frame["position"].mean()),
        "absolute_exposure": float(frame["position"].abs().mean()),
        "turnover": float(frame["position_change"].sum()),
        "trade_count": float((frame["position_change"] > 0).sum()),
        "observations": float(len(frame)),
        "long_signals": float(long_count),
        "short_signals": float(short_count),
        "abstain_signals": float((~active).sum()),
        "active_coverage": float(active.mean()),
        "active_directional_accuracy": float(realized_direction_correct[active].mean()) if active_count else np.nan,
        "long_hit_rate": float((frame.loc[long_active, "asset_return"] > 0).mean()) if long_count else np.nan,
        "short_hit_rate": float((frame.loc[short_active, "asset_return"] < 0).mean()) if short_count else np.nan,
        "long_total_return_unlevered": float(frame["long_strategy_return"].sum()),
        "short_total_return_unlevered": float(frame["short_strategy_return"].sum()),
    }
    return frame, metrics
