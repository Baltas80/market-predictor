import pandas as pd
import pytest

from market_predictor.historical_gate import validate_historical_dataset
from market_predictor.historical_ingestion import build_source_manifest


def market_frame():
    index = pd.DatetimeIndex([pd.Timestamp("2024-01-02 21:00:00+00:00"), pd.Timestamp("2024-01-03 21:00:00+00:00")])
    return pd.DataFrame({"open": [100.0, 101.0], "high": [101.0, 102.0], "low": [99.0, 100.0], "close": [100.5, 101.5], "volume": [1000.0, 1100.0]}, index=index)


def fred_macro_frame():
    return pd.DataFrame(
        {
            "series_id": ["DGS10", "DGS10"],
            "observation_date": ["2024-01-01", "2024-01-01"],
            "value": [3.90, 4.10],
            "vintage_start": ["2024-01-02", "2024-02-01"],
            "vintage_end": ["2024-01-31", "9999-12-31"],
        }
    )


def test_historical_gate_requires_manifest():
    with pytest.raises(RuntimeError, match="source manifest"):
        validate_historical_dataset(market_frame())


def test_historical_gate_admits_valid_dataset_with_manifest():
    market = market_frame()
    manifest = build_source_manifest(market, source_id="test-market", source_type="market", source_uri="test://market")
    result = validate_historical_dataset(market, manifests=[manifest])
    assert result["status"] == "admissible"
    assert result["market_observations"] == 2


def test_historical_gate_accepts_multiple_fred_vintages():
    market = market_frame()
    manifest = build_source_manifest(market, source_id="test-market", source_type="market", source_uri="test://market")
    result = validate_historical_dataset(market, macro=fred_macro_frame(), manifests=[manifest])
    assert result["status"] == "admissible"
    assert result["macro_rows"] == 2


def test_historical_gate_rejects_invalid_fred_vintage_interval():
    market = market_frame()
    macro = fred_macro_frame()
    macro.loc[0, "vintage_end"] = "2024-01-01"
    manifest = build_source_manifest(market, source_id="test-market", source_type="market", source_uri="test://market")
    with pytest.raises(ValueError, match="vintage_end cannot precede vintage_start"):
        validate_historical_dataset(market, macro=macro, manifests=[manifest])


def test_historical_gate_rejects_duplicate_fred_vintage():
    market = market_frame()
    macro = pd.concat([fred_macro_frame(), fred_macro_frame().iloc[[0]]], ignore_index=True)
    manifest = build_source_manifest(market, source_id="test-market", source_type="market", source_uri="test://market")
    with pytest.raises(ValueError, match="duplicate FRED vintage keys"):
        validate_historical_dataset(market, macro=macro, manifests=[manifest])
