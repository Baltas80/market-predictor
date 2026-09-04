import pandas as pd
import pytest

from market_predictor.events import build_event_features, merge_market_events, validate_events


def test_event_before_availability_is_rejected():
    events = pd.DataFrame(
        {
            "event_time": ["2026-01-02T00:00:00Z"],
            "available_time": ["2026-01-01T00:00:00Z"],
            "event_type": ["war"],
            "intensity": [1.0],
        }
    )
    with pytest.raises(ValueError):
        validate_events(events)


def test_future_available_event_cannot_leak_into_prediction():
    events = pd.DataFrame(
        {
            "event_time": ["2026-01-02T00:00:00Z"],
            "available_time": ["2026-01-03T12:00:00Z"],
            "event_type": ["war"],
            "intensity": [10.0],
        }
    )
    dates = pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"], utc=True)
    features = build_event_features(events, dates)
    assert features.loc["2026-01-02", "events_1d_count"] == 0.0
    assert features.loc["2026-01-03", "events_1d_count"] == 0.0
    assert features.loc["2026-01-04", "events_1d_count"] == 0.0
    assert features.loc["2026-01-04", "event_pressure"] > 0.0


def test_merge_preserves_market_rows():
    market = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True),
    )
    events = pd.DataFrame(
        {
            "event_time": ["2026-01-02T00:00:00Z"],
            "available_time": ["2026-01-02T00:00:00Z"],
            "event_type": ["armed_conflict"],
            "intensity": [2.0],
        }
    )
    features = build_event_features(events, market.index)
    merged = merge_market_events(market, features)
    assert list(merged["close"]) == [100.0, 101.0]
    assert merged["event_pressure"].iloc[0] > 0.0
