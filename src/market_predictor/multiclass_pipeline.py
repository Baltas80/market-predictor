"""Experimental expanding-window pipeline with PIT-safe temporal folds."""
from __future__ import annotations

import pandas as pd

from .directional_target import (
    DEFAULT_FLAT_RETURN_THRESHOLD,
    make_directional_target,
)
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

DEFAULT_EMBARGO = pd.Timedelta(days=1)


def prepare_multiclass_data(
    df: pd.DataFrame,
    horizon: int = 5,
    flat_return_threshold: float = DEFAULT_FLAT_RETURN_THRESHOLD,
) -> pd.DataFrame:
    """Build features/labels while retaining original label-end timestamps."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("multiclass pipeline requires a DatetimeIndex")
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("multiclass pipeline requires a unique, sorted DatetimeIndex")

    data = add_market_features(df).copy()
    data["label_end_time"] = pd.Series(
        df.index.to_series(index=df.index).shift(-horizon), index=df.index
    )
    data["direction_target"] = make_directional_target(
        data,
        horizon=horizon,
        flat_return_threshold=flat_return_threshold,
    )
    return data.dropna(subset=FEATURE_COLUMNS + ["direction_target", "label_end_time"]).copy()


def _make_pit_folds(
    data: pd.DataFrame,
    *,
    initial_train_size: int,
    test_size: int,
    embargo: pd.Timedelta,
) -> list[tuple[pd.Index, pd.Index, dict[str, str]]]:
    """Create folds using timestamps rather than row-count purge."""
    folds = []
    train_anchor = initial_train_size
    while train_anchor + test_size <= len(data):
        test = data.iloc[train_anchor : train_anchor + test_size]
        test_start = test.index[0]
        test_end = test.index[-1]
        train_candidates = data.iloc[:train_anchor]
        train = train_candidates[
            (train_candidates["label_end_time"] < test_start)
            & (train_candidates.index < test_start - embargo)
        ]
        if train.empty:
            raise ValueError("PIT fold has no admissible training observations")
        audit = {
            "train_start": train.index[0].isoformat(),
            "train_end": train.index[-1].isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "purge_rule": "label_end_time < test_start",
            "embargo": str(embargo),
            "embargo_rule": "train_timestamp < test_start - embargo",
        }
        folds.append((train.index, test.index, audit))
        train_anchor += test_size
    return folds


def run_multiclass_baseline(
    df: pd.DataFrame,
    *,
    horizon: int = 5,
    initial_train_fraction: float = 0.6,
    test_fraction: float = 0.1,
    flat_return_threshold: float = DEFAULT_FLAT_RETURN_THRESHOLD,
    embargo: pd.Timedelta = DEFAULT_EMBARGO,
) -> tuple[pd.DataFrame, list]:
    """Run expanding-window OOS prediction of DOWN/FLAT/UP.

    Every prediction carries the exact train/test timestamp audit information.
    The OOS lockbox is not modified by this experimental pipeline.
    """
    if not 0.5 <= initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must be >= 0.5 and < 1")
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be > 0 and < 0.5")
    if embargo < pd.Timedelta(0):
        raise ValueError("embargo must be >= 0")

    data = prepare_multiclass_data(
        df,
        horizon=horizon,
        flat_return_threshold=flat_return_threshold,
    )
    initial_train_size = max(1, int(len(data) * initial_train_fraction))
    test_size = max(1, int(len(data) * test_fraction))
    folds = _make_pit_folds(
        data,
        initial_train_size=initial_train_size,
        test_size=test_size,
        embargo=embargo,
    )

    prediction_parts: list[pd.DataFrame] = []
    evaluations: list = []
    for fold_id, (train_index, test_index, audit) in enumerate(folds):
        train = data.loc[train_index]
        test = data.loc[test_index]
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
        part["fold_id"] = fold_id
        for key, value in audit.items():
            part[key] = value
        prediction_parts.append(part)

    columns = [
        "prob_down",
        "prob_flat",
        "prob_up",
        "actual_direction",
        "predicted_direction",
        "fold_id",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "purge_rule",
        "embargo",
        "embargo_rule",
    ]
    if not prediction_parts:
        return pd.DataFrame(columns=columns), evaluations
    return pd.concat(prediction_parts).sort_index()[columns], evaluations
