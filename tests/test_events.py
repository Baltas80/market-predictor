import pandas as pd
import pytest

from market_predictor.events import build_event_features, merge_market_events


def _events(**overrides):
    data = {
        "event_time": ["2025-01-01T09:00:00Z"],
        "available_time": ["2025-01-01T12:00:00Z"],
        "event_type": ["war"],
        "intensity": [10.0],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_available_time_blocks_future_information():
    times = pd.date_range("2025-01-01T08:00:00Z", periods=6, freq="2h")
    features = build_event_features(_events(), times, windows=(1,))
    assert features.loc[times[0], "events_1d_count"] == 0
    assert features.loc[times[1], "events_1d_count"] == 0
    assert features.loc[times[2], "events_1d_count"] == 1
    assert features.loc[times[2], "event_pressure"] == pytest.approx(10.0)


def test_pressure_decays_from_availability_time():
    times = pd.to_datetime([
        "2025-01-01T12:00:00Z",
        "2025-01-06T12:00:00Z",
    ])
    features = build_event_features(_events(), times, windows=(10,))
    assert features.loc[times[0], "event_pressure"] == pytest.approx(10.0)
    assert features.loc[times[1], "event_pressure"] == pytest.approx(10.0 * 2.718281828 ** -1, rel=1e-5)


def test_conflict_taxonomy_includes_verbal_conflict():
    events = _events(event_type=["verbal_conflict"], available_time=["2025-01-01T09:00:00Z"])
    features = build_event_features(events, pd.to_datetime(["2025-01-01T09:00:00Z"]), windows=(1,))
    assert features.iloc[0]["events_1d_conflict_count"] == 1


def test_merge_preserves_market_rows():
    index = pd.date_range("2025-01-01", periods=2, tz="UTC")
    market = pd.DataFrame({"close": [100.0, 101.0]}, index=index)
    features = build_event_features(pd.DataFrame(columns=["event_time", "available_time", "event_type", "intensity"]), index)
    merged = merge_market_events(market, features)
    assert list(merged.index) == list(index)
    assert list(merged["close"]) == [100.0, 101.0]
