import pandas as pd
import pytest

from market_predictor.comparison import compare_predictions, prediction_index_hash


def results():
    idx = pd.date_range("2025-01-01", periods=5, tz="UTC")
    protocol = {
        "oos_start": "2025-01-01",
        "oos_end": "2025-01-05",
        "purge_gap": 5,
        "transaction_cost_bps": 5.0,
        "slippage_bps": 2.0,
        "observations": len(idx),
        "prediction_index_hash": prediction_index_hash(idx),
    }
    return {
        name: {"accuracy": 0.5, "total_return": 0.1, **protocol}
        for name in ("A", "B", "C", "D", "buy_and_hold", "random")
    }


def test_comparison_requires_all_protocol_members():
    table = compare_predictions(results())
    assert list(table.index) == ["A", "B", "C", "D", "buy_and_hold", "random"]


def test_comparison_refuses_missing_experiment():
    data = results()
    del data["D"]
    with pytest.raises(ValueError):
        compare_predictions(data)


def test_comparison_refuses_protocol_mismatch():
    data = results()
    data["C"]["purge_gap"] = 4
    with pytest.raises(ValueError, match="protocol mismatch"):
        compare_predictions(data)


def test_prediction_index_hash_is_deterministic_and_validates_index():
    idx = pd.date_range("2025-01-01", periods=3, tz="UTC")
    assert prediction_index_hash(idx) == prediction_index_hash(idx.copy())
    with pytest.raises(ValueError):
        prediction_index_hash(pd.DatetimeIndex([idx[1], idx[0]]))
