"""Fixed financial evaluation matrix for untouched OOS predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .financial import backtest_long_only

COST_BPS = (0.0, 5.0, 10.0)
SLIPPAGE_BPS = (0.0, 5.0)


def _random_probabilities(size: int, seed: int) -> np.ndarray:
    if size <= 0:
        raise ValueError("size must be positive")
    return np.random.default_rng(seed).random(size)


def evaluate_financial_matrix(
    predictions: dict[str, pd.DataFrame],
    *,
    benchmark: pd.DataFrame,
    probability_column: str = "prob_up",
    threshold: float = 0.5,
    periods_per_year: int = 252,
    costs_bps: tuple[float, ...] = COST_BPS,
    slippage_bps: tuple[float, ...] = SLIPPAGE_BPS,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Evaluate fixed OOS predictions across a pre-declared 3x2 cost grid.

    Predictions are consumed as-is. This function never retrains, tunes a
    threshold, changes the lockbox or selects a winner. Buy-and-hold and a
    deterministic random signal are reference strategies only.
    """
    if not predictions:
        raise ValueError("at least one prediction set is required")
    if not costs_bps or not slippage_bps:
        raise ValueError("cost and slippage grids cannot be empty")
    if "close" not in benchmark.columns:
        raise ValueError("benchmark must contain close")
    reference_index = benchmark.index
    for name, frame in predictions.items():
        if probability_column not in frame.columns or "close" not in frame.columns:
            raise ValueError(f"{name} must contain {probability_column} and close")
        if not frame.index.equals(reference_index):
            raise ValueError("all experiments and benchmark must use the same OOS index")

    random_frame = benchmark[["close"]].copy()
    random_frame[probability_column] = _random_probabilities(len(random_frame), random_seed)
    all_predictions = {**predictions, "random": random_frame}

    rows: list[dict[str, float | str]] = []
    for experiment_name, frame in all_predictions.items():
        for cost in costs_bps:
            for slippage in slippage_bps:
                _, metrics = backtest_long_only(
                    frame[[probability_column, "close"]],
                    probability_column=probability_column,
                    threshold=threshold,
                    transaction_cost_bps=float(cost),
                    slippage_bps=float(slippage),
                    periods_per_year=periods_per_year,
                )
                rows.append({
                    "experiment": experiment_name,
                    "transaction_cost_bps": float(cost),
                    "slippage_bps": float(slippage),
                    **metrics,
                })

    close = benchmark["close"].astype(float)
    asset_returns = close.shift(-1).dropna()
    buy_hold_total = float((1.0 + asset_returns).prod() - 1.0)
    result = pd.DataFrame(rows)
    result["buy_and_hold_total_return"] = buy_hold_total
    result["benchmark"] = "buy_and_hold"
    return result.sort_values(["experiment", "transaction_cost_bps", "slippage_bps"]).reset_index(drop=True)


def period_stability(
    backtest: pd.DataFrame,
    *,
    periods: dict[str, tuple[str, str]],
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Recompute fixed-backtest metrics over pre-declared chronological periods."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    rows = []
    for name, (start, end) in periods.items():
        subset = backtest.loc[(backtest.index >= pd.Timestamp(start)) & (backtest.index <= pd.Timestamp(end))]
        if subset.empty:
            rows.append({"period": name, "observations": 0})
            continue
        returns = subset["strategy_return"].astype(float)
        equity = (1.0 + returns).cumprod()
        std = returns.std(ddof=1)
        downside = (returns.clip(upper=0.0) ** 2).mean() ** 0.5
        rows.append({
            "period": name,
            "start": subset.index.min(),
            "end": subset.index.max(),
            "observations": len(subset),
            "total_return": float(equity.iloc[-1] - 1.0),
            "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
            "sharpe": float(returns.mean() / std * np.sqrt(periods_per_year)) if len(returns) > 1 and std > 0 else float("nan"),
            "sortino": float(returns.mean() / downside * np.sqrt(periods_per_year)) if len(returns) > 1 and downside > 0 else float("nan"),
            "turnover": float(subset["position_change"].sum()),
        })
    return pd.DataFrame(rows)
