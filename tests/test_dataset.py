from __future__ import annotations

import json

import pandas as pd
import pytest

from market_predictor.dataset import build_manifest, load_events, load_fred_vintage, load_market, materialize_macro


def test_load_market_rejects_duplicate_dates(tmp_path) -> None:
    path = tmp_path / "market.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01"],
            "open": [1, 1], "high": [1, 1], "low": [1, 1],
            "close": [1, 1], "volume": [1, 1],
        }
    ).to_csv(path, index=False)
    try:
        load_market(path)
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("duplicate market dates were accepted")


def test_materialize_macro_preserves_only_point_in_time_values(tmp_path) -> None:
    path = tmp_path / "fred.csv"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "value": [1.0, 2.0],
            "realtime_start": pd.to_datetime(["2020-01-01", "2020-01-03"]),
        }
    ).to_csv(path, index=False)
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    from market_predictor.data_sources import align_fred_point_in_time
    result = materialize_macro(index, {"TEST": path}, align_fred_point_in_time)
    assert pd.isna(result.loc["2020-01-01", "TEST"])
    assert result.loc["2020-01-04", "TEST"] == 2.0


def test_load_fred_vintage_accepts_missing_realtime_end(tmp_path) -> None:
    path = tmp_path / "fred.csv"
    pd.DataFrame({
        "date": ["2020-01-01"],
        "value": [1.0],
        "realtime_start": ["2020-01-02"],
    }).to_csv(path, index=False)
    result = load_fred_vintage(path)
    assert len(result) == 1
    assert "realtime_end" not in result.columns


def test_load_fred_vintage_requires_realtime_start(tmp_path) -> None:
    path = tmp_path / "fred.csv"
    pd.DataFrame({"date": ["2020-01-01"], "value": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="realtime_start"):
        load_fred_vintage(path)


def test_load_events_deduplicates_ids(tmp_path) -> None:
    path = tmp_path / "events.csv"
    pd.DataFrame(
        {
            "event_id": ["1", "1"], "category": ["war_conflict"] * 2,
            "published_at": ["2020-01-01T12:00:00Z"] * 2,
            "severity": [0.5, 0.5], "country": ["US", "US"],
            "entity": ["A", "A"], "sector": [None, None],
            "duration_days": [0, 0], "media_intensity": [1, 1], "surprise": [0, 0],
        }
    ).to_csv(path, index=False)
    assert len(load_events([path])) == 1


def test_manifest_contains_sha256(tmp_path) -> None:
    path = tmp_path / "input.txt"
    path.write_text("abc", encoding="utf-8")
    manifest = build_manifest([path], dataset_name="test", start="2020-01-01", end="2020-01-02")
    assert manifest["files"][0]["sha256"]
    json.dumps(manifest)
