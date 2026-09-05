import pytest

from market_predictor.comparison import compare_predictions


def protocol():
    return {"oos_start":"2025-01-01", "oos_end":"2025-12-31", "purge_gap":5, "transaction_cost_bps":5.0, "slippage_bps":2.0}


def test_comparison_requires_all_models_and_common_protocol():
    p = protocol()
    results = {name: {**p, "accuracy": 0.5} for name in ("A","B","C","D","buy_and_hold","random")}
    table = compare_predictions(results)
    assert list(table.index) == ["A","B","C","D","buy_and_hold","random"]


def test_comparison_rejects_mixed_protocol():
    p = protocol()
    results = {name: {**p, "accuracy": 0.5} for name in ("A","B","C","D","buy_and_hold","random")}
    results["D"]["purge_gap"] = 10
    with pytest.raises(ValueError):
        compare_predictions(results)
