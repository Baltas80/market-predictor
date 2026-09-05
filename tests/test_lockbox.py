from datetime import date

import pytest

from market_predictor.lockbox import LockboxSpec, result_artifact_hash


def spec():
    return LockboxSpec(date(2024,1,1), date(2024,12,31), 5, "dataset", "code", "protocol-1", 5.0, 2.0)


def test_lockbox_fingerprint_is_stable():
    assert spec().fingerprint(model_configs={"seed": 42}) == spec().fingerprint(model_configs={"seed": 42})
    assert spec().fingerprint(model_configs={"seed": 42}) != spec().fingerprint(model_configs={"seed": 43})


def test_lockbox_rejects_invalid_spec():
    bad = LockboxSpec(date(2025,1,1), date(2024,1,1), 5, "dataset", "code", "protocol", 0, 0)
    with pytest.raises(ValueError):
        bad.validate()


def test_result_hash_is_order_independent_for_mappings():
    assert result_artifact_hash({"a": 1, "b": 2}) == result_artifact_hash({"b": 2, "a": 1})
