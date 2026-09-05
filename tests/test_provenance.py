from datetime import datetime, timezone

import pytest

from market_predictor.provenance import DatasetProvenance


def test_provenance_accepts_valid_record():
    record = DatasetProvenance("market", "stooq", datetime(2026, 1, 1, tzinfo=timezone.utc), "a" * 64, "1")
    assert record.as_dict()["dataset_name"] == "market"


def test_provenance_rejects_naive_timestamp():
    record = DatasetProvenance("market", "stooq", datetime(2026, 1, 1), "a" * 64, "1")
    with pytest.raises(ValueError, match="timezone-aware"):
        record.validate()


def test_provenance_rejects_invalid_hash():
    record = DatasetProvenance("market", "stooq", datetime(2026, 1, 1, tzinfo=timezone.utc), "not-a-hash", "1")
    with pytest.raises(ValueError, match="SHA-256"):
        record.validate()
