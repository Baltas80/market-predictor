import pandas as pd

from market_predictor.event_features import events_to_features
from market_predictor.event_schema import EventCategory, MarketEvent


def test_event_features_never_use_future_events():
    index = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
    event = MarketEvent("c1", EventCategory.CORRUPTION, "2025-01-03T12:00:00Z", 1.0)
    features = events_to_features(index, [event])
    assert features.loc[index[0], "event_corruption"] == 0
    assert features.loc[index[1], "event_corruption"] == 0
    assert features.loc[index[2], "event_corruption"] == 0
    assert features.loc[index[3], "event_corruption"] > 0


def test_event_features_preserve_event_category_signal():
    index = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
    event = MarketEvent("s1", EventCategory.CORPORATE_SCANDAL, index[0], 0.7, surprise=0.5)
    features = events_to_features(index, [event])
    assert features.loc[index[0], "event_corporate_scandal"] == 0.7
    assert features.loc[index[0], "event_total_pressure"] == 0.7
    assert features.loc[index[0], "event_surprise"] == 0.35
