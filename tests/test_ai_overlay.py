import pandas as pd
import pytest

from market_predictor.ai_overlay import assert_d_uses_c_lockbox, prepare_ai_overlay


def test_ai_signals_are_admitted_at_first_eligible_decision():
    idx = pd.DatetimeIndex(pd.to_datetime(["2025-01-02 21:00Z", "2025-01-03 21:00Z", "2025-01-06 21:00Z"], utc=True))
    c = pd.DataFrame({"prob_up": [0.5, 0.6, 0.4]}, index=idx)
    signals = pd.DataFrame({"available_at": pd.to_datetime(["2025-01-03 12:00Z", "2025-01-07 00:00Z"], utc=True), "score": [1.0, 2.0]})
    result = prepare_ai_overlay(c, signals)
    assert len(result) == 1
    assert result.iloc[0]["first_eligible_decision"] == idx[1]


def test_ai_overlay_rejects_naive_c_index():
    c = pd.DataFrame({"prob_up": [0.5]}, index=pd.DatetimeIndex(["2025-01-02 21:00"]))
    signals = pd.DataFrame({"available_at": pd.to_datetime(["2025-01-02 12:00Z"], utc=True)})
    with pytest.raises(ValueError, match="timezone-aware"):
        prepare_ai_overlay(c, signals)


def test_d_must_preserve_c_index():
    idx = pd.date_range("2025-01-01", periods=2, tz="UTC")
    c = pd.DataFrame(index=idx)
    d = pd.DataFrame(index=idx)
    assert_d_uses_c_lockbox(c, d)
    with pytest.raises(ValueError, match="exactly C"):
        assert_d_uses_c_lockbox(c, d.iloc[:1])
