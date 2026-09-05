"""Fixed financial evaluation matrix for untouched OOS predictions."""

from __future__ import annotations

import pandas as pd

from .benchmarks import random_signal
from .financial import backtest_long_only

COST_BPS = (0.0, 5.0, 10.0)
SLIPPAGE_BPS = (0.0, 5.0)


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
    """Evaluate fixed OOS predictions across a pre-declared cost/slippage grid.

    No model is retrained and no threshold is selected from the resulting
    table. The random baseline is generated once on the common prediction
    index, making it a comparator rather than a tuning candidate.
    """
    if not predictions:
        raise ValueError("at least one prediction set is required")
    if not costs_bps or not slippage_bps:
        raise ValueError("cost and slippage grids cannot be empty")
    reference_index = None
    for name, frame in predictions.items():
        if probability_column not in frame.columns:
            raise ValueError(f"{name} missing {probability_column}")
        if reference_index is None:
            reference_index = frame.index
        elif not frame.index.equals(reference_index):
            raise ValueError("all experiments must use the same OOS index")
    if probability_column not in benchmark.columns or "close" not in benchmark.columns:
        raise ValueError("benchmark must contain probability and close columns")
    if not benchmark.index.equals(reference_index):
        raise ValueError("benchmark must use the same OOS index")

    random_frame = benchmark[["close"]].copy()
    random_frame[probability_column] = random_signal(len(random_frame), seed=random_seed)
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

    # Buy-and-hold is a fixed, model-free comparator. It is reported once
    # against the common OOS close series rather than duplicated per model.
    close = benchmark["close"].astype(float)
    asset_returns = close.shift(-1).dropna()
    buy_hold_total = float((1.0 + asset_returns).prod() - 1.0)
    for row in rows:
        row["buy_and_hold_total_return"] = buy_hold_total
    result = pd.DataFrame(rows)
    return result.sort_values(["experiment", "transaction_cost_bps", "slippage_bps"]).reset_index(drop=True)


def period_stability(
    backtest: pd.DataFrame,
    *,
    periods: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Recompute financial metrics over pre-declared chronological periods."""
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
            "sharpe": float(returns.mean() / std * (252 ** 0.5)) if len(returns) > 1 and std > 0 else float("nan"),
            "sortino": float(returns.mean() / downside * (252 ** 0.5)) if len(returns) > 1 and downside > 0 else float("nan"),
            "turnover": float(subset["position_change"].sum()),
        })
    return pd.DataFrame(rows)
