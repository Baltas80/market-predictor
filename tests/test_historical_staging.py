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
            "pit_start": "2024-01-02",
            "last_vintage_in_range": end,
        }
    }
    return frame


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


def test_fred_vintage_windows_bound_long_requests():
    windows = _fred_vintage_windows("2000-01-03", "2025-12-31")
    assert windows[0] == ("2000-01-03", "2001-01-01")
    assert windows[-1][1] == "2025-12-31"
    assert all(
        pd.Timestamp(end) - pd.Timestamp(start) <= pd.Timedelta(days=364)
        for start, end in windows
    )


def test_fred_vintage_dates_are_discovered_before_observation_requests(monkeypatch):
    captured = {}
    payload = {"vintage_dates": ["2005-06-28", "2014-04-02", "2025-01-07"]}

    def fake_get(*args, **kwargs):
        captured.update(kwargs.get("params", {}))
        return SimpleNamespace(json=lambda: payload)

    monkeypatch.setattr(historical_adapters, "_get", fake_get)
    result = _fred_vintage_dates("test-key", "DGS10", "2000-01-03", "2025-12-31")

    assert captured["series_id"] == "DGS10"
    assert captured["realtime_start"] == "2000-01-03"
    assert captured["realtime_end"] == "2025-12-31"
    assert captured["limit"] == 10000
    assert result[0] == pd.Timestamp("2005-06-28", tz="UTC")
    assert result[-1] == pd.Timestamp("2025-01-07", tz="UTC")


def test_fetch_fred_starts_each_series_at_first_valid_pit_vintage(monkeypatch):
    requested_windows = []

    monkeypatch.setattr(
        historical_adapters,
        "_fred_vintage_dates",
        lambda api_key, series_id, start, end: pd.DatetimeIndex([pd.Timestamp("2005-06-28", tz="UTC")])
        if series_id == "DGS10"
        else pd.DatetimeIndex([pd.Timestamp("2000-01-03", tz="UTC")]),
    )

    def fake_observations(series_id, api_key, *, realtime_start=None, realtime_end=None):
        requested_windows.append((series_id, realtime_start, realtime_end))
        return pd.DataFrame(
            {
                "series_id": [series_id],
                "date": pd.to_datetime([realtime_start], utc=True),
                "value": [1.0],
                "realtime_start": pd.to_datetime([realtime_start], utc=True),
                "realtime_end": pd.to_datetime([realtime_end], utc=True),
            }
        )

    monkeypatch.setattr(historical_adapters, "load_fred_observations", fake_observations)
    result = historical_adapters.fetch_fred("test-key", "2000-01-03", "2000-12-31")

    assert result.attrs["fred_pit_coverage"]["DGS10"]["pit_start"] == "2005-06-28"
    assert all(window[0] != "DGS10" for window in requested_windows)


def test_staging_fails_before_download_without_fred_credentials(tmp_path: Path):
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        stage_historical(
            tmp_path,
            fred_api_key=None,
            include_gdelt=False,
            include_sec=False,
            market_fetcher=_market,
            fred_fetcher=_macro,
        )
