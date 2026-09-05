from market_predictor.reproducibility import canonical_json_hash, run_fingerprint


def test_canonical_hash_is_independent_of_mapping_order():
    assert canonical_json_hash({"a": 1, "b": 2}) == canonical_json_hash({"b": 2, "a": 1})


def test_run_fingerprint_changes_when_inputs_change():
    base = run_fingerprint(dataset_hash="dataset", config={"seed": 42}, code_version="code")
    assert base != run_fingerprint(dataset_hash="dataset-2", config={"seed": 42}, code_version="code")
    assert base != run_fingerprint(dataset_hash="dataset", config={"seed": 43}, code_version="code")
    assert base != run_fingerprint(dataset_hash="dataset", config={"seed": 42}, code_version="code-2")


def test_run_fingerprint_requires_dataset_and_code_version():
    import pytest

    with pytest.raises(ValueError):
        run_fingerprint(dataset_hash="", config={}, code_version="code")
    with pytest.raises(ValueError):
        run_fingerprint(dataset_hash="dataset", config={}, code_version="")
