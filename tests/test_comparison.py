import pytest

from market_predictor.comparison import compare_predictions


def results():
    return {name: {"accuracy": 0.5, "total_return": 0.1} for name in ("A", "B", "C", "D", "buy_and_hold", "random")}


def test_comparison_requires_all_protocol_members():
    table = compare_predictions(results())
    assert list(table.index) == ["A", "B", "C", "D", "buy_and_hold", "random"]


def test_comparison_refuses_missing_experiment():
    data = results(); del data["D"]
    with pytest.raises(ValueError):
        compare_predictions(data)
