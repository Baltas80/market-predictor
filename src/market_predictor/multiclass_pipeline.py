"""Experimental expanding-window pipeline for exhaustive direction prediction."""
from __future__ import annotations

import pandas as pd

from .backtest import Fold, make_walk_forward_folds
from .directional_target import make_directional_target
from .features import add_market_features
from .multiclass_model import evaluate_multiclass, fit_predict_multiclass


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "volatility_20d",
    "price_to_sma20",
    "volume_change",
    "range_pct",
]


def prepare_multiclass_data(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Build leakage-safe market features and exhaustive future-direction labels."""
    data = add_market_features(df)
    data["direction_target"] = make_directional_target(data, horizon=horizon)
    return data.dropna(subset=FEATURE_COLUMNS + ["direction_target"]).copy()


def run_multiclass_baseline(
    df: pd.DataFrame,
    *,
    horizon: int = 5,
    initial_train_fraction: float = 0.6,
    test_fraction: float = 0.1,
) -> tuple[pd.DataFrame, list]:
    """Run expanding-window OOS prediction of DOWN/FLAT/UP."""
    if not 0.5 <= initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must be >= 0.5 and < 1")
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be > 0 and < 0.5")

    data = prepare_multiclass_data(df, horizon=horizon)
    initial_train_size = max(1, int(len(data) * initial_train_fraction))
    test_size = max(1, int(len(data) * test_fraction))
    folds = make_walk_forward_folds(
        len(data),
        initial_train_size=initial_train_size,
        test_size=test_size,
        purge=horizon,
    )

    prediction_parts: list[pd.DataFrame] = []
    evaluations: list = []
    for fold in folds:
        train = data.iloc[fold.train_start:fold.train_end]
        test = data.iloc[fold.test_start:fold.test_end]
        _, probabilities, predicted = fit_predict_multiclass(
            train[FEATURE_COLUMNS],
            train["direction_target"],
            test[FEATURE_COLUMNS],
        )
        evaluations.append(
            evaluate_multiclass(test["direction_target"], probabilities, predicted)
        )
        part = probabilities.copy()
        part["actual_direction"] = test["direction_target"].astype(int)
        part["predicted_direction"] = predicted
        prediction_parts.append(part)

    if not prediction_parts:
        return (
            pd.DataFrame(
                columns=[
                    "prob_down",
                    "prob_flat",
                    "prob_up",
                    "actual_direction",
                    "predicted_direction",
                ]
            ),
            evaluations,
        )
    return pd.concat(prediction_parts).sort_index(), evaluations
