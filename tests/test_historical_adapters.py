from __future__ import annotations

import pandas as pd
import pytest

from market_predictor import historical_adapters


def _event_frame(event_id: str = "1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [event_id],
            "event_time": pd.to_datetime(["2020-01-01"], utc=True),
            "published_at": pd.NaT,
            "available_at": pd.to_datetime(["2020-01-01T12:00:00Z"], utc=True),
            "category": ["war_conflict"],
            "severity": [0.5],
            "country": ["US"],
            "entity": ["ACTOR"],
            "sector": [pd.NA],
            "duration_days": [0.0],
            "media_intensity": [1.0],
            "surprise": [0.0],
            "source": ["GDELT_2_Event_Database"],
            "availability_proxy": ["DATEADDED"],
            "source_url": ["https://example.test/1"],
        }
    )


def test_fetch_gdelt_retries_transient_daily_failure(monkeypatch, capsys) -> None:
    calls = {"count": 0}

    def fake_load(day):
        calls["count"] += 1
        if calls["count"] == 1:
            raise historical_adapters.requests.RequestException("temporary")
        return pd.DataFrame()

    monkeypatch.setattr(historical_adapters, "load_gdelt_day", fake_load)
    monkeypatch.setattr(historical_adapters.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        historical_adapters,
        "gdelt_events_to_market_events",
        lambda frame: _event_frame(),
    )
    monkeypatch.setattr(historical_adapters, "deduplicate_events", lambda frame: frame)
    monkeypatch.setattr(historical_adapters, "normalize_event_sources", lambda frame, source_id: frame)

    result = historical_adapters.fetch_gdelt("2020-01-01", "2020-01-01")

    assert calls["count"] == 2
    assert len(result) == 1
    assert "GDELT staging progress: 1/1 days" in capsys.readouterr().out


def test_fetch_gdelt_fails_with_day_in_error_after_retries(monkeypatch) -> None:
    def fake_load(day):
        raise historical_adapters.requests.RequestException("offline")

    monkeypatch.setattr(historical_adapters, "load_gdelt_day", fake_load)
    monkeypatch.setattr(historical_adapters.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="2020-01-01"):
        historical_adapters.fetch_gdelt("2020-01-01", "2020-01-01")
