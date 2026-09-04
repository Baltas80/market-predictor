import numpy as np
import pandas as pd
import pytest

from market_predictor.pipeline import prepare_baseline_data, run_final_lockbox


def _ohlcv(n=140):
    rng = np.random.default_rng(7)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    index = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * (1 + rng.uniform(0, 0.01, n)),
            "low": close * (1 - rng.uniform(0, 0.01, n)),
            "close": close,
            "volume": rng.integers(1000, 10000, n),
        },
        index=index,
    )


def test_final_lockbox_returns_only_one_untouched_test_block():
    raw = _ohlcv()
    prepared = prepare_baseline_data(raw, horizon=5)
    result, evaluations = run_final_lockbox(raw, horizon=5, test_fraction=0.2)
    expected_test_size = max(1, int(len(prepared) * 0.2))
    assert len(evaluations) == 1
    assert len(result) == expected_test_size
    assert result.index.is_monotonic_increasing
    assert result.index.is_unique


def test_lockbox_rejects_insufficient_data():
    with pytest.raises(ValueError, match="not enough observations"):
        run_final_lockbox(_ohlcv(30), horizon=20, test_fraction=0.4)


def test_lockbox_requires_valid_test_fraction():
    with pytest.raises(ValueError, match="test_fraction"):
        run_final_lockbox(_ohlcv(), test_fraction=0.5)
