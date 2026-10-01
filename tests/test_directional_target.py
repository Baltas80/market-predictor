import pandas as pd
import pytest

from market_predictor.directional_target import make_directional_target


def test_directional_target_exhaustively_covers_up_down_and_flat():
    index = pd.date_range("2025-01-01", periods=5, tz="UTC")
    frame = pd.DataFrame({"close": [100.0, 101.0, 100.0, 100.0, 102.0]}, index=index)
    target = make_directional_target(frame, horizon=1)
    assert target.iloc[0] == 1
    assert target.iloc[1] == -1
    assert target.iloc[2] == 0
    assert target.iloc[3] == 1
    assert pd.isna(target.iloc[4])


def test_directional_target_rejects_invalid_horizon():
    frame = pd.DataFrame({"close": [100.0, 101.0]})
    with pytest.raises(ValueError):
        make_directional_target(frame, horizon=0)


def test_directional_target_never_fabricates_last_label():
    frame = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    target = make_directional_target(frame, horizon=2)
    assert target.iloc[-2:].isna().all()
