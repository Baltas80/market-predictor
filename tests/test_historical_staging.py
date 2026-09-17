from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from market_predictor import historical_adapters
from market_predictor.historical_adapters import _fred_vintage_dates, _fred_vintage_windows
from market_predictor.historical_staging import stage_historical


def _market(start: str, end: str) -> pd.DataFrame:
    index = pd.DatetimeIndex([
        pd.Timestamp("2024-01-02 21:00:00+00:00"),
        pd.Timestamp("2024-01-03 21:00:00+00:00"),
    ])
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1100.0],
        },
        index=index,
    )


def _macro(api_key: str, start: str, end: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "series_id": ["FEDFUNDS", "FEDFUNDS"],
            "observation_date": pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True),
            "value": [5.0, 5.1],
            "vintage_start": pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True),
            "vintage_end": pd.to_datetime(["2024-01-03", "2024-01-04"], utc=True),
        }
    )
    frame.attrs["fred_pit_coverage"] = {
        "FEDFUNDS": {
            "requested_start": start,
            "pit_start": start,
            "first_vintage": start,
            "last_vintage_in_history": end,
        }
    }
    return frame


def _gdelt(source_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["g1"],
            "event_time": pd.to_datetime(["2024-01-02"], utc=True),
            "published_at": [pd.NaT],
            "available_at": pd.to_datetime(["2024-01-03T11:00:00Z"], utc=True),
            "source_id": [source_id],
            "category": ["political_crisis"],
            "severity": [0.5],
            "country": [None],
            "entity": [None],
            "sector": [None],
            "duration_days": [0.0],
            "media_intensity": [1.0],
            "surprise": [0.0],
        }
    )


def test_staging_runs_all_phases_and_writes_manifest(tmp_path: Path):
    result = stage_historical(
        tmp_path,
        fred_api_key="test-key",
        include_gdelt=False,
        include_sec=False,
        market_fetcher=_market,
        fred_fetcher=_macro,
    )

    assert result["status"] == "admissible"
    assert [stage["status"] for stage in result["stages"]] == [
        "complete", "complete", "complete", "complete", "complete", "complete", "admissible"
    ]
    assert (tmp_path / "raw" / "market.csv").exists()
    assert (tmp_path / "raw" / "macro_fred.csv").exists()
    assert (tmp_path / "normalized" / "market.csv").exists()
    assert (tmp_path / "normalized" / "macro.csv").exists()
    assert (tmp_path / "source_manifest.json").exists()
    assert (tmp_path / "staging_result.json").exists()


def test_staging_records_yahoo_fallback_provenance(tmp_path: Path):
    market = _market("", "").copy()
    market.attrs["source_id"] = "YahooFinance_GSPC"

    result = stage_historical(
        tmp_path,
        fred_api_key="test-key",
        include_gdelt=False,
        include_sec=False,
        market_fetcher=lambda start, end: market,
        fred_fetcher=_macro,
    )

    assert result["status"] == "admissible_with_source_limits"
    assert any("Yahoo Finance" in item for item in result["limitations"])
    manifest_text = (tmp_path / "source_manifest.json").read_text(encoding="utf-8")
    assert "YahooFinance_GSPC" in manifest_text


def test_staging_rejects_noncanonical_gdelt_source_id(tmp_path: Path):
    with pytest.raises(RuntimeError, match="non-canonical GDELT source identifiers"):
        stage_historical(
            tmp_path,
            fred_api_key="test-key",
            include_gdelt=True,
            include_sec=False,
            market_fetcher=_market,
            fred_fetcher=_macro,
            gdelt_fetcher=lambda start, end: _gdelt("GDELT_2_Event_Database"),
        )


def test_staging_accepts_canonical_gdelt_source_id(tmp_path: Path):
    result = stage_historical(
        tmp_path,
        fred_api_key="test-key",
        include_gdelt=True,
        include_sec=False,
        market_fetcher=_market,
        fred_fetcher=_macro,
        gdelt_fetcher=lambda start, end: _gdelt("GDELT_1_Event_Database"),
    )
    assert result["historical_gate"] is not None
    normalized = pd.read_csv(tmp_path / "normalized" / "events_gdelt.csv")
    assert set(normalized.columns) >= {
        "event_id", "event_time", "published_at", "available_at", "source_id",
        "category", "severity", "country", "entity", "sector", "duration_days",
        "media_intensity", "surprise",
    }
    assert normalized.loc[0, "source_id"] == "GDELT_1_Event_Database"


def test_fred_vintage_windows_bound_long_requests():
    windows = _fred_vintage_windows("2000-01-03", "2025-12-31")
    assert windows[0] == ("2000-01-03", "2001-01-01")
    assert windows[-1][1] == "2025-12-31"
    assert all(
        pd.Timestamp(end) - pd.Timestamp(start) <= pd.Timedelta(364, unit="D")
        for start, end in windows
    )


def test_fred_vintage_dates_are_paginated_and_sorted(monkeypatch):
    captured = []
    payloads = [
        {"count": 3, "vintage_dates": ["2005-06-28", "2014-04-02"]},
        {"count": 3, "vintage_dates": ["2025-01-07"]},
    ]

    def fake_get(*args, **kwargs):
        captured.append(kwargs.get("params", {}).copy())
        return SimpleNamespace(json=lambda: payloads[len(captured) - 1])

    monkeypatch.setattr(historical_adapters, "_get", fake_get)
    result = _fred_vintage_dates("test-key", "DGS10")
