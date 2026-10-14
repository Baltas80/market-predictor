"""Regression tests for research-specific purged walk-forward folds."""

import pytest

from market_predictor.backtest import Fold, make_walk_forward_folds


def test_folds_are_chronological_and_expanding():
    folds = make_walk_forward_folds(
        n_rows=30,
        initial_train_size=10,
        test_size=4,
        step=4,
        purge=2,
    )

    assert folds == [
        Fold(0, 10, 12, 16),
        Fold(0, 14, 16, 20),
        Fold(0, 18, 20, 24),
        Fold(0, 22, 24, 28),
    ]
    for previous, current in zip(folds, folds[1:]):
        assert current.train_end > previous.train_end
        assert current.test_start >= previous.test_end


def test_purge_is_applied_between_training_and_test():
    folds = make_walk_forward_folds(
        n_rows=20,
        initial_train_size=8,
        test_size=3,
        step=3,
        purge=5,
    )

    assert folds
    assert all(fold.test_start - fold.train_end == 5 for fold in folds)


def test_zero_purge_is_supported_for_explicit_non_overlapping_targets():
    folds = make_walk_forward_folds(
        n_rows=12,
        initial_train_size=6,
        test_size=3,
        purge=0,
    )

    assert folds == [Fold(0, 6, 6, 9), Fold(0, 9, 9, 12)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_rows": 0, "initial_train_size": 1, "test_size": 1}, "positive"),
        ({"n_rows": 10, "initial_train_size": 0, "test_size": 1}, "positive"),
        ({"n_rows": 10, "initial_train_size": 2, "test_size": 0}, "positive"),
        ({"n_rows": 10, "initial_train_size": 2, "test_size": 1, "purge": -1}, "purge"),
        ({"n_rows": 10, "initial_train_size": 2, "test_size": 1, "step": 0}, "step"),
    ],
)
def test_invalid_fold_parameters_fail_closed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_walk_forward_folds(**kwargs)


def test_insufficient_rows_returns_no_fold_instead_of_leaking_future_rows():
    folds = make_walk_forward_folds(
        n_rows=10,
        initial_train_size=8,
        test_size=3,
        purge=2,
    )

    assert folds == []
