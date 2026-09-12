import pandas as pd

from scripts.ingest_gdelt2_bigquery import _normalize


def test_gdelt2_dateadded_is_availability_and_not_event_time():
    frame = pd.DataFrame(
        [
            {
                "GLOBALEVENTID": "123",
                "SQLDATE": "20260910",
                "Actor1Name": "A",
                "Actor1CountryCode": "ESP",
                "Actor2Name": "B",
                "Actor2CountryCode": "USA",
                "EventCode": "190",
                "EventRootCode": "19",
                "GoldsteinScale": "-5",
                "NumMentions": "10",
                "NumSources": "3",
                "NumArticles": "4",
                "AvgTone": "-2.5",
                "ActionGeo_CountryCode": "ESP",
                "ActionGeo_FullName": "Spain",
                "DATEADDED": "20260911121500",
                "SOURCEURL": "https://example.test/article",
            }
        ]
    )
    normalized = _normalize(frame)
    assert len(normalized) == 1
    row = normalized.iloc[0]
    assert row["event_time"] == pd.Timestamp("2026-09-10", tz="UTC")
    assert row["available_at"] == pd.Timestamp("2026-09-11 12:15:00", tz="UTC")
    assert row["available_at"] > row["event_time"]
    assert row["event_id"] == "123"


def test_gdelt2_rejects_availability_before_event_time():
    frame = pd.DataFrame(
        [
            {
                "GLOBALEVENTID": "124",
                "SQLDATE": "20260912",
                "Actor1Name": "A",
                "Actor1CountryCode": "ESP",
                "Actor2Name": "B",
                "Actor2CountryCode": "USA",
                "EventCode": "190",
                "EventRootCode": "19",
                "GoldsteinScale": "-5",
                "NumMentions": "10",
                "NumSources": "3",
                "NumArticles": "4",
                "AvgTone": "-2.5",
                "ActionGeo_CountryCode": "ESP",
                "ActionGeo_FullName": "Spain",
                "DATEADDED": "20260911121500",
                "SOURCEURL": "https://example.test/article",
            }
        ]
    )
    assert _normalize(frame).empty
