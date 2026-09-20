import pandas as pd
import pytest

from market_predictor.directional_target import make_directional_target


def test_directional_target_uses_frozen_neutrality_band():
    index = pd.date_range("2025-01-01", periods=5, tz="UTC")
    frame = pd.DataFrame(
        {"close": [100.0, 100.05, 100.0, 99.0, 102.0]}, index=index
    )
    target = make_directional_target(frame, horizon=1, flat_return_threshold=0.001)
    assert target.iloc[0] == 0  # +5 bps => FLAT
    assert target.iloc[1] == 0  # -5 bps => FLAT
    assert target.iloc[2] == -1  # -100 bps => DOWN
    assert target.iloc[3] == 1  # +303 bps => UP
    assert pd.isna(target.iloc[4])


def test_directional_target_rejects_invalid_parameters():
    frame = pd.DataFrame({"close": [100.0, 101.0]})
    with pytest.raises(ValueError):
        make_directional_target(frame, horizon=0)
    with pytest.raises(ValueError):
        make_directional_target(frame, flat_return_threshold=-0.1)


def test_directional_target_never_fabricates_last_label():
    frame = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    target = make_directional_target(frame, horizon=2)
    assert target.iloc[-2:].isna().all()
