"""Sensitivity analysis for fixed out-of-sample financial predictions."""

from __future__ import annotations

from itertools import product

import pandas as pd

from .financial import backtest_long_only


METRIC_COLUMNS = (
    "total_return",
    "cagr",
    "max_drawdown",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "hit_rate",
    "average_position",
    "turnover",
)


def run_sensitivity_grid(
    predictions: pd.DataFrame,
    *,
    thresholds: list[float] | tuple[float, ...] = (0.45, 0.50, 0.55, 0.60),
    transaction_costs_bps: list[float] | tuple[float, ...] = (0.0, 5.0, 10.0),
    slippages_bps: list[float] | tuple[float, ...] = (0.0, 5.0),
    probability_column: str = "prob_up",
    close_column: str = "close",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Evaluate every parameter combination without retraining the model.

    The supplied predictions are treated as a fixed OOS lockbox. Each row is
    an independent financial simulation over exactly the same observations.
    """
    thresholds = tuple(float(x) for x in thresholds)
    transaction_costs_bps = tuple(float(x) for x in transaction_costs_bps)
    slippages_bps = tuple(float(x) for x in slippages_bps)
    if not thresholds or not transaction_costs_bps or not slippages_bps:
        raise ValueError("sensitivity grids cannot be empty")

    rows: list[dict[str, float]] = []
    for threshold, cost, slippage in product(
        thresholds, transaction_costs_bps, slippages_bps
    ):
        _, metrics = backtest_long_only(
            predictions,
            probability_column=probability_column,
            close_column=close_column,
            threshold=threshold,
            transaction_cost_bps=cost,
            slippage_bps=slippage,
            periods_per_year=periods_per_year,
        )
        row = {
            "threshold": threshold,
            "transaction_cost_bps": cost,
            "slippage_bps": slippage,
        }
        row.update({name: metrics[name] for name in METRIC_COLUMNS})
        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=[
            "threshold",
            "transaction_cost_bps",
            "slippage_bps",
            *METRIC_COLUMNS,
        ],
    )
