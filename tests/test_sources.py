import pandas as pd
import pytest

from market_predictor.sources.gdelt import normalize_gdelt_events
from market_predictor.sources.market_csv import load_ohlcv_csv
from market_predictor.sources.fred import FredClient


def test_gdelt_normalization_maps_core_fields():
    raw = pd.DataFrame(
        {
            "GlobalEventID": [123],
            "SQLDATE": ["20240115"],
            "DATEADDED": ["20240115123000"],
            "EventCode": [190],
            "QuadClass": [4],
            "GoldsteinScale": [-7.0],
            "NumMentions": [9],
            "NumSources": [3],
            "AvgTone": [-4.2],
        }
    )
    out = normalize_gdelt_events(raw)
    assert out.loc[0, "event_type"] == "material_conflict"
    assert out.loc[0, "global_event_id"] == 123
    assert out.loc[0, "available_time"] == pd.Timestamp("2024-01-15 12:30:00", tz="UTC")
    assert out.loc[0, "intensity"] > 0


def test_gdelt_normalization_accepts_exact_numeric_dateadded_encodings():
    raw = pd.DataFrame(
        {
            "GlobalEventID": [123, 124],
            "SQLDATE": ["20240115", "20240115"],
            "DATEADDED": ["20240115123000.0", "2.024011513e13"],
            "EventCode": [190, 190],
            "QuadClass": [4, 4],
            "GoldsteinScale": [-7.0, -7.0],
            "NumMentions": [9, 9],
            "NumSources": [3, 3],
            "AvgTone": [-4.2, -4.2],
        }
    )
    out = normalize_gdelt_events(raw)
    assert out["available_time"].tolist() == [
        pd.Timestamp("2024-01-15 12:30:00", tz="UTC"),
        pd.Timestamp("2024-01-15 13:00:00", tz="UTC"),
    ]


def test_gdelt_normalization_rejects_non_integral_dateadded():
    raw = pd.DataFrame(
        {
            "GlobalEventID": [123],
            "SQLDATE": ["20240115"],
            "DATEADDED": ["20240115123000.9"],
            "EventCode": [190],
            "QuadClass": [4],
            "GoldsteinScale": [-7.0],
            "NumMentions": [9],
            "NumSources": [3],
            "AvgTone": [-4.2],
        }
    )
    with pytest.raises(ValueError, match="timestamps are invalid"):
        normalize_gdelt_events(raw)


def test_market_csv_rejects_invalid_ohlcv(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01,10,9,8,10,100\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="high is below"):
        load_ohlcv_csv(path)


def test_fred_requires_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        FredClient().observations("CPIAUCSL")
