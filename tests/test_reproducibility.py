import pytest

from market_predictor.reproducibility import canonical_json_hash, run_fingerprint


def test_canonical_json_hash_is_order_independent_for_dicts():
    left = {"b": 2, "a": [1, 2]}
    right = {"a": [1, 2], "b": 2}
    assert canonical_json_hash(left) == canonical_json_hash(right)


def test_run_fingerprint_changes_when_dataset_config_or_code_changes():
    base = run_fingerprint(dataset_hash="abc", config={"horizon": 5}, code_version="v1")
    assert base != run_fingerprint(dataset_hash="def", config={"horizon": 5}, code_version="v1")
    assert base != run_fingerprint(dataset_hash="abc", config={"horizon": 10}, code_version="v1")
    assert base != run_fingerprint(dataset_hash="abc", config={"horizon": 5}, code_version="v2")


def test_run_fingerprint_requires_dataset_and_code_version():
    with pytest.raises(ValueError, match="required"):
        run_fingerprint(dataset_hash="", config={}, code_version="v1")
    with pytest.raises(ValueError, match="required"):
        run_fingerprint(dataset_hash="abc", config={}, code_version="")
