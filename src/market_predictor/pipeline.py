"""Chronological, purged walk-forward baseline pipeline."""

from __future__ import annotations

import pandas as pd

from .backtest import make_walk_forward_folds, walk_forward_classification
from .features import add_market_features, make_target

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "volatility_20d",
    "price_to_sma20",
    "volume_change",
    "range_pct",
]


def prepare_baseline_data(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Build model-ready features and remove rows whose target is unknown."""
    data = add_market_features(df)
    data["target"] = make_target(data, horizon=horizon)
    return data.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()


def run_baseline(
    df: pd.DataFrame,
    horizon: int = 5,
    initial_train_fraction: float = 0.6,
    test_fraction: float = 0.1,
) -> tuple[pd.DataFrame, list]:
    """Run expanding-window out-of-sample evaluation with a purge gap.

    The gap equals ``horizon`` so training labels cannot reach into the first
    observations of the test window.
    """
    if not 0.5 <= initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must be >= 0.5 and < 1")
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be > 0 and < 0.5")

    data = prepare_baseline_data(df, horizon=horizon)
    initial_train_size = max(1, int(len(data) * initial_train_fraction))
    test_size = max(1, int(len(data) * test_fraction))
    folds = make_walk_forward_folds(
        len(data),
        initial_train_size=initial_train_size,
        test_size=test_size,
        purge=horizon,
    )
    return walk_forward_classification(data, FEATURE_COLUMNS, "target", folds)
