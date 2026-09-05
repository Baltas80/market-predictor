from __future__ import annotations

import pandas as pd

from market_predictor.data_sources import _gdelt_category, gdelt_events_to_market_events


def test_gdelt_category_uses_exact_sanctions_code() -> None:
    assert _gdelt_category("16", "163", "UNITED STATES RUSSIA") == "sanctions"


def test_gdelt_category_keeps_unknown_event_broad() -> None:
    assert _gdelt_category("01", "010", "ACTOR") == "political_crisis"


def test_gdelt_event_conversion_deduplicates_and_preserves_availability() -> None:
    frame = pd.DataFrame(
        {
            "global_event_id": ["1", "1", "2"],
            "category": ["war_conflict", "war_conflict", "social_unrest"],
            "date_added": pd.to_datetime(
                ["2020-01-01T12:00:00Z", "2020-01-01T12:00:00Z", "2020-01-02T12:00:00Z"],
                utc=True,
            ),
            "severity": [0.8, 0.8, 0.4],
            "action_geo_country": ["US", "US", "FR"],
            "actor2_name": ["ACTOR", "ACTOR", "ACTOR2"],
            "num_sources": [5, 5, 2],
            "surprise": [0.0, 0.0, 0.0],
        }
    )
    result = gdelt_events_to_market_events(frame)
    assert len(result) == 2
    assert result.iloc[0]["published_at"] == "2020-01-01T12:00:00Z"
    assert result.iloc[0]["event_id"] == "1"
