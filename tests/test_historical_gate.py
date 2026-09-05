import pandas as pd
import pytest

from market_predictor.historical_gate import validate_historical_dataset
from market_predictor.historical_ingestion import build_source_manifest


def market_frame():
    index = pd.DatetimeIndex([pd.Timestamp("2024-01-02 21:00:00+00:00"), pd.Timestamp("2024-01-03 21:00:00+00:00")])
    return pd.DataFrame({"open": [100.0, 101.0], "high": [101.0, 102.0], "low": [99.0, 100.0], "close": [100.5, 101.5], "volume": [1000.0, 1100.0]}, index=index)


def test_historical_gate_requires_manifest():
    with pytest.raises(RuntimeError, match="source manifest"):
        validate_historical_dataset(market_frame())


def test_historical_gate_admits_valid_dataset_with_manifest():
    market = market_frame()
    manifest = build_source_manifest(market, source_id="test-market", source_type="market", source_uri="test://market")
    result = validate_historical_dataset(market, manifests=[manifest])
    assert result["status"] == "admissible"
    assert result["market_observations"] == 2
