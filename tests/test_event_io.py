import pandas as pd
import pytest

from market_predictor.event_io import load_events_csv


def test_load_events_csv_parses_and_sorts(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {"event_id": "b", "category": "corruption", "published_at": "2025-01-03T10:00:00Z", "severity": 0.8},
            {"event_id": "a", "category": "corporate_scandal", "published_at": "2025-01-01T10:00:00Z", "severity": 0.6, "surprise": 0.4},
        ]
    ).to_csv(path, index=False)
    events = load_events_csv(path)
    assert [event.event_id for event in events] == ["a", "b"]
    assert events[0].surprise == 0.4


def test_load_events_csv_rejects_duplicates(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {"event_id": "x", "category": "corruption", "published_at": "2025-01-01", "severity": 0.5},
            {"event_id": "x", "category": "corruption", "published_at": "2025-01-02", "severity": 0.5},
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate event_id"):
        load_events_csv(path)


def test_load_events_csv_rejects_unknown_category(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame([{"event_id": "x", "category": "made_up", "published_at": "2025-01-01", "severity": 0.5}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Invalid event"):
        load_events_csv(path)
