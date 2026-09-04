"""Time-series backtesting utilities with a purge gap."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .model import Evaluation, evaluate, fit_predict


@dataclass(frozen=True)
class Fold:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def make_walk_forward_folds(
    n_rows: int,
    initial_train_size: int,
    test_size: int,
    step: int | None = None,
    purge: int = 0,
) -> list[Fold]:
    """Create expanding-window folds with a gap between train and test.

    ``purge`` should normally be at least the prediction horizon when labels
    use future observations. This prevents training labels near the boundary
    from incorporating information from the test period.
    """
    if n_rows <= 0 or initial_train_size < 1 or test_size < 1:
        raise ValueError("n_rows, initial_train_size and test_size must be positive")
    if purge < 0:
        raise ValueError("purge must be >= 0")
    step = test_size if step is None else step
    if step < 1:
        raise ValueError("step must be >= 1")

    folds: list[Fold] = []
    train_end = initial_train_size
    while train_end + purge + test_size <= n_rows:
        test_start = train_end + purge
        folds.append(Fold(0, train_end, test_start, test_start + test_size))
        train_end += step
    return folds


def walk_forward_classification(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    folds: list[Fold],
) -> tuple[pd.DataFrame, list[Evaluation]]:
    """Fit one fresh model per fold and return out-of-sample predictions."""
    predictions: list[pd.Series] = []
    evaluations: list[Evaluation] = []

    for fold in folds:
        train = data.iloc[fold.train_start : fold.train_end]
        test = data.iloc[fold.test_start : fold.test_end]
        model, probabilities = fit_predict(
            train[features], train[target].astype(int), test[features]
        )
        del model
        evaluations.append(evaluate(test[target].astype(int), probabilities))
        predictions.append(
            pd.DataFrame(
                {"actual": test[target].astype(int), "prob_up": probabilities},
                index=test.index,
            )
        )

    if not predictions:
        return pd.DataFrame(columns=["actual", "prob_up"]), []
    return pd.concat(predictions).sort_index(), evaluations
