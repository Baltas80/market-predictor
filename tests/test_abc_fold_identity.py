from datetime import date

import pandas as pd
import pytest

from market_predictor.abc_execution_guard import build_abc_execution_plan
from market_predictor.abc_protocol import ABCProtocol, folds_identity_hash
from market_predictor.backtest import Fold
from market_predictor.reproducibility import canonical_json_hash


def _protocol(index: pd.Index, folds: tuple[Fold, ...]) -> ABCProtocol:
    prediction_index = index[folds[0].test_start:folds[0].test_end]
    return ABCProtocol(
        lockbox_start=date(2024, 1, 1),
        lockbox_end=date(2024, 1, 31),
        purge_gap=2,
        horizon=2,
        transaction_cost_bps=5.0,
        slippage_bps=0.0,
        observations=len(index),
        prediction_index_hash=canonical_json_hash([ts.isoformat() for ts in prediction_index]),
        dataset_hash="dataset",
        code_version="test",
        folds_hash=folds_identity_hash(folds),
    )


def test_abc_plan_rejects_fold_boundary_mutation_even_when_oos_index_is_unchanged():
    index = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    original = (Fold(0, 3, 5, 7),)
    mutated = (Fold(0, 2, 5, 7),)
    protocol = _protocol(index, original)

    plan = build_abc_execution_plan(protocol=protocol, folds=original, observation_index=index)
    assert plan.folds == original

    with pytest.raises(ValueError, match="folds_hash"):
        build_abc_execution_plan(protocol=protocol, folds=mutated, observation_index=index)


def test_fold_identity_hash_changes_when_any_boundary_changes():
    first = (Fold(0, 3, 5, 7), Fold(0, 5, 7, 8))
    second = (Fold(0, 4, 5, 7), Fold(0, 5, 7, 8))
    assert folds_identity_hash(first) != folds_identity_hash(second)
