from pathlib import Path

import pandas as pd
import pytest

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
    return pd.DataFrame(
        {
            "series_id": ["DFF", "DFF"],
            "observation_date": pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True),
            "value": [5.0, 5.1],
            "vintage_start": pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True),
            "vintage_end": pd.to_datetime(["2024-01-03", "2024-01-04"], utc=True),
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
    assert (tmp_path / "raw" / "market_stooq.csv").exists()
    assert (tmp_path / "raw" / "macro_fred.csv").exists()
    assert (tmp_path / "normalized" / "market.csv").exists()
    assert (tmp_path / "normalized" / "macro.csv").exists()
    assert (tmp_path / "source_manifest.json").exists()
    assert (tmp_path / "staging_result.json").exists()


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
