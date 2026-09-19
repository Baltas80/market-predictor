import pandas as pd
import pytest

from market_predictor.directional import backtest_directional, directional_position


def test_probability_maps_to_long_short_and_abstain():
    probabilities = pd.Series([0.9, 0.5, 0.1], index=pd.date_range("2025-01-01", periods=3, tz="UTC"))
    result = directional_position(probabilities)
    assert result.tolist() == [1, 0, -1]


def test_directional_backtest_uses_next_period_return_for_both_sides():
    index = pd.date_range("2025-01-01", periods=4, tz="UTC")
    frame = pd.DataFrame(
        {"prob_up": [0.9, 0.1, 0.9, 0.1], "close": [100.0, 110.0, 100.0, 90.0]},
        index=index,
    )
    result, metrics = backtest_directional(frame, transaction_cost_bps=0, slippage_bps=0)
    assert result.iloc[0]["strategy_return"] == pytest.approx(0.10)
    assert result.iloc[1]["strategy_return"] == pytest.approx(0.10)
    assert metrics["long_signals"] == 2
    assert metrics["short_signals"] == 1


def test_directional_backtest_charges_reversal_as_two_units_of_turnover():
    index = pd.date_range("2025-01-01", periods=4, tz="UTC")
    frame = pd.DataFrame(
        {"prob_up": [0.9, 0.1, 0.5, 0.5], "close": [100.0, 100.0, 100.0, 100.0]},
        index=index,
    )
    _, metrics = backtest_directional(frame, transaction_cost_bps=10, slippage_bps=0)
    assert metrics["turnover"] == pytest.approx(4.0)
    assert metrics["trade_count"] == pytest.approx(3.0)


def test_thresholds_must_define_a_real_abstain_band():
    with pytest.raises(ValueError):
        directional_position(pd.Series([0.5]), upper_threshold=0.5, lower_threshold=0.4)
    with pytest.raises(ValueError):
        directional_position(pd.Series([0.5]), upper_threshold=0.6, lower_threshold=0.5)
