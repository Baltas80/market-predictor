from datetime import date

import pandas as pd

from market_predictor.gdelt1 import GDELT_SOURCE_ID, gdelt1_daily_availability
from scripts.ingest_gdelt_daily_chunk import _availability_for_file_date, load_gdelt_daily_day


def test_gdelt1_source_uses_canonical_next_day_boundary():
    available = _availability_for_file_date(date(2015, 2, 19))
    assert available == gdelt1_daily_availability(date(2015, 2, 19))
    assert available == pd.Timestamp("2015-02-20T11:00:00Z")
    assert GDELT_SOURCE_ID == "GDELT_1_Event_Database"


def test_gdelt1_source_handles_daylight_saving_time():
    available = _availability_for_file_date(date(2015, 7, 1))
    assert available == pd.Timestamp("2015-07-02T10:00:00Z")


def test_legacy_daily_loader_is_backed_by_canonical_source(monkeypatch):
    raw = pd.DataFrame(
        {
            "global_event_id": ["g1"],
            "sql_date": pd.to_datetime(["2015-02-19"], utc=True),
            "available_at": pd.to_datetime(["2015-02-20T11:00:00Z"], utc=True),
            "category": ["political_crisis"],
            "severity": [0.5],
        }
    )
    monkeypatch.setattr("scripts.ingest_gdelt_daily_chunk.load_gdelt_day", lambda day: raw.copy())
    frame = load_gdelt_daily_day("2015-02-19")
    assert frame.loc[0, "source_id"] == GDELT_SOURCE_ID
    assert frame.loc[0, "availability_proxy"] == "daily_archive_publication_boundary"
    assert pd.isna(frame.loc[0, "published_at"])
