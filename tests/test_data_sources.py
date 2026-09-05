from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from market_predictor import data_sources
from market_predictor.data_sources import (
    _gdelt_category,
    _sec_category,
    align_fred_point_in_time,
    gdelt_events_to_market_events,
    load_stooq_daily,
)
from market_predictor.research_schema import normalize_event_sources, validate_event_frame


def test_gdelt_category_uses_exact_sanctions_code() -> None:
    assert _gdelt_category("16", "163", "UNITED STATES RUSSIA") == "sanctions"


def test_gdelt_category_keeps_unknown_event_broad() -> None:
    assert _gdelt_category("01", "010", "ACTOR") == "political_crisis"


def test_gdelt_event_conversion_deduplicates_and_preserves_availability() -> None:
    frame = pd.DataFrame(
        {
            "global_event_id": ["1", "1", "2"],
            "sql_date": pd.to_datetime(["2020-01-01"] * 2 + ["2020-01-02"], utc=True),
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
            "source_url": ["https://example.test/1"] * 2 + ["https://example.test/2"],
        }
    )
    result = gdelt_events_to_market_events(frame)
    assert len(result) == 2
    assert pd.isna(result.iloc[0]["published_at"])
    assert result.iloc[0]["available_at"] == pd.Timestamp("2020-01-01T12:00:00Z")
    assert result.iloc[0]["event_time"] == pd.Timestamp("2020-01-01T00:00:00Z")
    assert result.iloc[0]["event_id"] == "1"
    assert result.iloc[0]["source"] == "GDELT_2_Event_Database"
    assert result.iloc[0]["availability_proxy"] == "DATEADDED"


def test_unknown_publication_is_valid_when_availability_is_known() -> None:
    event = normalize_event_sources(
        pd.DataFrame(
            {
                "event_id": ["g1"],
                "event_time": pd.to_datetime(["2020-01-01"], utc=True),
                "published_at": [pd.NaT],
                "available_at": pd.to_datetime(["2020-01-01T12:00:00Z"], utc=True),
            }
        ),
        source_id="GDELT_2_Event_Database",
    )
    validate_event_frame(event)


def test_fred_alignment_does_not_use_same_day_vintage_by_default() -> None:
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "value": [1.0, 2.0],
            "realtime_start": pd.to_datetime(["2020-01-01", "2020-01-03"]),
        }
    )
    market_index = pd.date_range("2020-01-01", periods=4, freq="D")
    aligned = align_fred_point_in_time(observations, market_index)
    assert pd.isna(aligned.loc["2020-01-01", "value"])
    assert aligned.loc["2020-01-04", "value"] == 2.0


def test_sec_category_distinguishes_fraud_and_scandal() -> None:
    assert _sec_category("Accounting fraud charges", "") == "financial_fraud"
    assert _sec_category("Insider trading case", "") == "corporate_scandal"
    assert _sec_category("Commission order", "New regulatory action") == "regulation"


def test_stooq_accepts_lowercase_date_header(monkeypatch) -> None:
    response = SimpleNamespace(
        text="date,open,high,low,close,volume\n2020-01-02,3200,3220,3190,3210,1000000\n"
    )
    monkeypatch.setattr(data_sources, "_get", lambda *args, **kwargs: response)
    result = load_stooq_daily("^spx", start="2020-01-02", end="2020-01-02")
    assert len(result) == 1
    assert result.iloc[0]["close"] == 3210.0
    assert result.index[0].tzinfo is not None
