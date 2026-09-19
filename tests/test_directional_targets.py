import pandas as pd
import pytest

from market_predictor.directional_targets import make_directional_target


def test_explicit_up_down_neutral_target():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    data = pd.DataFrame({"close": [100.0, 100.0, 100.0, 101.0, 99.0, 100.0]}, index=idx)

    target = make_directional_target(data, horizon=1, neutral_threshold=0.005)

    assert target.iloc[0] == 0.0
    assert target.iloc[2] == 1.0
    assert target.iloc[3] == -1.0
    assert pd.isna(target.iloc[-1])


def test_directional_target_zero_threshold_treats_flat_as_neutral():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    data = pd.DataFrame({"close": [100.0, 100.0, 101.0]}, index=idx)

    target = make_directional_target(data, horizon=1, neutral_threshold=0.0)

    assert target.iloc[0] == 0.0
    assert target.iloc[1] == 1.0


def test_directional_target_rejects_invalid_inputs():
    data = pd.DataFrame({"close": [100.0, 101.0]})
    with pytest.raises(ValueError):
        make_directional_target(data, horizon=0)
    with pytest.raises(ValueError):
        make_directional_target(data, neutral_threshold=-0.01)
    with pytest.raises(ValueError):
        make_directional_target(pd.DataFrame({"open": [1.0, 2.0]}))
