import numpy as np
import pandas as pd

from market_predictor.data_quality import check_market


def frame():
    idx = pd.date_range("2025-01-01", periods=2, tz="UTC")
    return pd.DataFrame({"open":[10,11],"high":[11,12],"low":[9,10],"close":[10.5,11.5],"volume":[100,120]}, index=idx)


def test_market_requires_all_ohlcv_columns():
    checks = check_market(frame().drop(columns="volume"))
    assert not next(c for c in checks if c.name == "market_columns").passed


def test_market_rejects_non_datetime_index():
    data = frame()
    data.index = ["2025-01-01", "2025-01-02"]
    checks = check_market(data)
    assert not next(c for c in checks if c.name == "market_index_datetime").passed


def test_market_rejects_nonpositive_prices():
    data = frame(); data.loc[data.index[0], "close"] = 0
    assert not next(c for c in check_market(data) if c.name == "market_prices_positive").passed


def test_market_rejects_negative_volume():
    data = frame(); data.loc[data.index[0], "volume"] = -1
    assert not next(c for c in check_market(data) if c.name == "market_volume_nonnegative").passed


def test_market_rejects_nonfinite_values():
    data = frame(); data.loc[data.index[0], "close"] = np.inf
    assert not next(c for c in check_market(data) if c.name == "market_finite").passed
