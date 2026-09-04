import numpy as np
import pandas as pd
import pytest

from market_predictor.financial import backtest_long_only


def test_signal_uses_next_period_return_and_charges_turnover():
    index = pd.date_range("2025-01-01", periods=5, tz="UTC")
    frame = pd.DataFrame(
        {"prob_up": [0.9, 0.9, 0.1, 0.1, 0.9], "close": [100, 110, 100, 100, 110]},
        index=index,
    )
    result, metrics = backtest_long_only(frame, transaction_cost_bps=0, slippage_bps=0)
    assert result.iloc[0]["strategy_return"] == pytest.approx(0.10)
    assert result.iloc[1]["strategy_return"] == pytest.approx(-0.0909090909)
    assert metrics["observations"] == 4


def test_threshold_controls_position():
    index = pd.date_range("2025-01-01", periods=3, tz="UTC")
    frame = pd.DataFrame({"prob_up": [0.55, 0.45, 0.55], "close": [100, 101, 102]}, index=index)
    result, _ = backtest_long_only(frame, threshold=0.6, transaction_cost_bps=0)
    assert np.all(result["signal"].to_numpy() == 0)
