import pandas as pd
import pytest

from market_predictor.fred_pit import audit_fred_point_in_time, fred_availability_cutoff
from market_predictor.historical_ingestion import assemble_common_observations, audit_event_timing
from market_predictor.information_set import build_information_set
from market_predictor.session_calendar import session_close


def _market(index):
    return pd.DataFrame({"open": [100.0] * len(index), "high": [101.0] * len(index), "low": [99.0] * len(index), "close": [100.0] * len(index), "volume": [1.0] * len(index)}, index=index)


def test_information_set_rejects_market_timezone_mismatch():
    market = _market(pd.DatetimeIndex(["2024-01-02 21:00:00+00:00"]))
    decision = pd.Timestamp("2024-01-02 16:00:00-05:00")
    with pytest.raises(ValueError, match="timezone"):
        build_information_set(decision_time=decision, market=market)


def test_information_set_accepts_explicitly_matching_timezone():
    close = pd.Timestamp("2024-01-02 16:00:00", tz="America/New_York")
    market = _market(pd.DatetimeIndex([close]))
    result = build_information_set(decision_time=close, market=market)
    assert result.market_row["close"] == 100.0


def test_fred_audit_rejects_invalid_vintage_order():
    frame = pd.DataFrame({
        "series_id": ["DGS10"], "date": ["2024-01-10"], "value": [4.0],
        "realtime_start": ["2024-01-11"], "realtime_end": ["2024-01-10"],
    })
    with pytest.raises(ValueError, match="realtime_end"):
        audit_fred_point_in_time(frame)


def test_fred_cutoff_is_explicit_and_conservative():
    decision = pd.Timestamp("2024-01-10 21:00:00+00:00")
    assert fred_availability_cutoff(decision, conservative_session_lag=1) == pd.Timestamp("2024-01-09 00:00:00+00:00")


def test_historical_assembly_rejects_non_session_market_index():
    saturday = pd.DatetimeIndex([pd.Timestamp("2024-01-06 21:00:00+00:00")])
    with pytest.raises(RuntimeError, match="session audit"):
        assemble_common_observations(_market(saturday))


def test_event_timing_uses_next_eligible_market_decision():
    first = pd.Timestamp("2024-01-02 21:00:00+00:00")
    second = pd.Timestamp("2024-01-03 21:00:00+00:00")
    events = pd.DataFrame({
        "event_id": ["e1"], "event_time": [pd.Timestamp("2024-01-02 12:00:00+00:00")],
        "published_at": [pd.NaT], "available_at": [pd.Timestamp("2024-01-03 12:00:00+00:00")],
        "source_id": ["test"], "category": ["political_crisis"], "severity": [0.5],
        "country": ["US"], "entity": ["x"], "sector": [pd.NA], "duration_days": [0.0],
        "media_intensity": [1.0], "surprise": [0.0],
    })
    audit = audit_event_timing(events, pd.DatetimeIndex([first, second]))
    assert audit.loc[0, "first_eligible_decision"] == second


def test_session_close_is_timezone_aware():
    close = session_close(pd.Timestamp("2024-01-02").date())
    assert close.tzinfo is not None
