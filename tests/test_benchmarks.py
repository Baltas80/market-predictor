import numpy as np
import pandas as pd
import pytest

from market_predictor.benchmarks import buy_and_hold_returns, random_signal


def test_buy_and_hold_returns_are_one_step_returns():
    index = pd.date_range("2025-01-01", periods=3, tz="UTC")
    close = pd.Series([100.0, 110.0, 99.0], index=index)
    result = buy_and_hold_returns(close)
    assert result.tolist() == pytest.approx([0.10, -0.10])
    assert result.index.equals(index[1:])


def test_buy_and_hold_rejects_non_positive_prices():
    close = pd.Series([100.0, 0.0, 101.0])
    with pytest.raises(ValueError, match="positive"):
        buy_and_hold_returns(close)


def test_random_signal_is_reproducible_and_index_aligned():
    index = pd.date_range("2025-01-01", periods=10, tz="UTC")
    probabilities = pd.Series(np.linspace(0.1, 0.9, 10), index=index)
    first = random_signal(probabilities, seed=7)
    second = random_signal(probabilities, seed=7)
    assert first.equals(second)
    assert first.index.equals(index)
    assert set(first.unique()).issubset({0, 1})


@pytest.mark.parametrize(
    "values",
    [
        [0.1, np.nan, 0.9],
        [0.1, np.inf, 0.9],
        [0.1, -0.01, 0.9],
        [0.1, 1.01, 0.9],
    ],
)
def test_random_signal_rejects_invalid_probability_values(values):
    probabilities = pd.Series(values)
    with pytest.raises(ValueError, match="probabilities"):
        random_signal(probabilities)
