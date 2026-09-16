import numpy as np
import pandas as pd
import pytest

from market_predictor.features import add_market_features, make_target


def sample_data(n=60):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series(range(100, 100 + n), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 1,
            "low": close - 2,
            "close": close,
            "volume": 1000.0,
        },
        index=idx,
    )


def test_features_are_created():
    out = add_market_features(sample_data())
    assert "return_1d" in out
    assert "volatility_20d" in out
    assert "price_to_sma20" in out


def test_target_horizon():
    out = sample_data()
    target = make_target(out, horizon=5)
    assert target.iloc[0] == 1
    assert pd.isna(target.iloc[-1])


def test_invalid_horizon():
    with pytest.raises(ValueError):
        make_target(sample_data(), horizon=0)


def test_non_finite_derived_features_are_quarantined():
    data = sample_data()
    data.loc[data.index[10], "volume"] = 0.0
    data.loc[data.index[11], "volume"] = 1000.0
    data.loc[data.index[12], "close"] = 0.0
    data.loc[data.index[12], "high"] = 1.0
    data.loc[data.index[12], "low"] = -1.0

    out = add_market_features(data)
    derived = [
        "return_1d", "return_5d", "volatility_20d", "sma_20", "sma_50",
        "price_to_sma20", "volume_change", "range_pct",
    ]
    assert np.isfinite(out[derived].dropna().to_numpy()).all()
    assert pd.isna(out.loc[data.index[11], "volume_change"])
    assert pd.isna(out.loc[data.index[12], "range_pct"])
