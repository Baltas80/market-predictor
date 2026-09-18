
from datetime import date

import pytest

from market_predictor.lockbox_manifest import LockboxManifest


def make_manifest(**overrides):
    payload = {
        "experiment_id": "exp-1234abcd5678",
        "git_commit": "a" * 40,
        "branch": "main",
        "dataset_hash": "d" * 64,
        "source_hashes": (("market", "m" * 64), ("macro", "x" * 64)),
        "coverage": {"observations": 100},
        "feature_hash": "f" * 64,
        "model": "StandardScaler + LogisticRegression",
        "hyperparameters": {"max_iter": 2000},
        "seeds": (42,),
        "fold_definition": ({"train_start": 0, "train_end": 80, "test_start": 85, "test_end": 100, "purge": 5},),
        "train_ranges": (("2020-01-01T00:00:00+00:00", "2020-03-31T00:00:00+00:00"),),
        "validation_ranges": (),
        "oos_ranges": (("2020-04-01T00:00:00+00:00", "2020-04-30T00:00:00+00:00"),),
        "purge_gap": 5,
        "embargo": 0,
        "transaction_cost_bps": 5.0,
        "slippage_bps": 0.0,
        "benchmark": "buy_and_hold_close",
        "software_version": "0.1.0",
        "python_version": "3.12.0",
        "dependencies": ("market-predictor==0.1.0", "numpy==2.0.0"),
        "result_hashes": (("financial_report", "r" * 64),),
        "artifact_paths": ("data/results/financial_report.json",),
        "execution_timestamp": "2026-09-18T12:00:00Z",
        "oos_start": date(2020, 4, 1),
        "oos_end": date(2020, 4, 30),
        "protocol_version": "protocol-v1",
        "observations": 100,
    }
    payload.update(overrides)
    return LockboxManifest(**payload)


def test_complete_manifest_validates_and_fingerprints():
    manifest = make_manifest()
    manifest.validate()
    assert len(manifest.fingerprint()) == 64
    serialized = manifest.as_dict()
    assert serialized["experiment_id"] == "exp-1234abcd5678"
    assert serialized["validation_ranges"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("experiment_id", ""),
        ("git_commit", "short"),
        ("branch", ""),
        ("dataset_hash", ""),
        ("feature_hash", ""),
        ("model", ""),
        ("benchmark", ""),
        ("software_version", ""),
        ("python_version", ""),
        ("execution_timestamp", "not-a-timestamp"),
    ],
)
def test_manifest_requires_core_provenance(field, value):
    with pytest.raises(ValueError):
        make_manifest(**{field: value}).validate()


def test_manifest_requires_sources_folds_ranges_dependencies_and_artifacts():
    for field, value in [
        ("source_hashes", ()),
        ("fold_definition", ()),
        ("train_ranges", ()),
        ("oos_ranges", ()),
        ("seeds", ()),
        ("dependencies", ()),
        ("result_hashes", ()),
        ("artifact_paths", ()),
    ]:
        with pytest.raises(ValueError):
            make_manifest(**{field: value}).validate()
