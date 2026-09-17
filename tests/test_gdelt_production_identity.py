from pathlib import Path

import pandas as pd

from market_predictor.historical_staging import stage_historical


def _market(start: str, end: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02 21:00:00+00:00")]),
    )


def _macro(api_key: str, start: str, end: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["FEDFUNDS"],
            "observation_date": pd.to_datetime(["2024-01-02"], utc=True),
            "value": [5.0],
            "vintage_start": pd.to_datetime(["2024-01-02"], utc=True),
            "vintage_end": pd.to_datetime(["2024-01-03"], utc=True),
        }
    )


def _gdelt(start: str, end: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["evt-1"],
            "event_time": pd.to_datetime(["2024-01-01"], utc=True),
            "published_at": pd.NaT,
            "available_at": pd.to_datetime(["2024-01-02T11:00:00Z"], utc=True),
            "source_id": ["GDELT_1_Event_Database"],
            "category": ["political_crisis"],
            "severity": [0.5],
            "country": ["US"],
            "entity": ["ACTOR"],
            "sector": [pd.NA],
            "duration_days": [0.0],
            "media_intensity": [1.0],
            "surprise": [0.0],
            "source": ["GDELT_1_Event_Database"],
            "availability_proxy": ["conservative_next_day_boundary"],
            "source_url": ["https://example.test/event"],
        }
    )


def test_historical_staging_uses_gdelt1_identity_and_pit_policy(tmp_path: Path) -> None:
    result = stage_historical(
        tmp_path,
        fred_api_key="test-key",
        include_gdelt=True,
        include_sec=False,
        market_fetcher=_market,
        fred_fetcher=_macro,
        gdelt_fetcher=_gdelt,
    )

    assert result["status"] == "admissible"
    manifest = (tmp_path / "source_manifest.json").read_text(encoding="utf-8")
    assert "GDELT_1_Event_Database" in manifest
    assert "GDELT_2_Event_Database" not in manifest
    assert "DATEADDED is retained as availability proxy" not in manifest
    assert "Conservative next-day 06:00 America/New_York publication boundary" in manifest
