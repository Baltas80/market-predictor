import pandas as pd
import pytest

from market_predictor.fred_pit import audit_fred_point_in_time, fred_availability_cutoff


def vintage_frame():
    return pd.DataFrame(
        {
            "series_id": ["TEST", "TEST"],
            "date": ["2024-01-01", "2024-01-01"],
            "value": [100.0, 101.0],
            "realtime_start": ["2024-01-02", "2024-02-01"],
            "realtime_end": ["2024-01-31", "9999-12-31"],
        }
    )


def test_fred_audit_accepts_multiple_vintages_for_one_observation():
    audited = audit_fred_point_in_time(vintage_frame())
    assert len(audited) == 2
    assert audited["realtime_start"].tolist() == [
        pd.Timestamp("2024-01-02", tz="UTC"),
        pd.Timestamp("2024-02-01", tz="UTC"),
    ]


def test_fred_vintage_before_decision_cutoff_is_not_future_revision():
    audited = audit_fred_point_in_time(vintage_frame())
    cutoff = fred_availability_cutoff("2024-01-15 21:00:00+00:00", conservative_session_lag=1)
    eligible = audited[audited["realtime_start"] <= cutoff]
    assert eligible["value"].tolist() == [100.0]


def test_fred_revision_becomes_eligible_only_after_its_vintage_date():
    audited = audit_fred_point_in_time(vintage_frame())
    cutoff = fred_availability_cutoff("2024-02-02 21:00:00+00:00", conservative_session_lag=1)
    eligible = audited[audited["realtime_start"] <= cutoff]
    assert eligible["value"].tolist() == [100.0, 101.0]


def test_fred_audit_accepts_realtime_start_before_observation_date():
    data = vintage_frame()
    data.loc[0, "realtime_start"] = "2023-12-31"
    audited = audit_fred_point_in_time(data)
    assert audited.loc[0, "realtime_start"] == pd.Timestamp("2023-12-31", tz="UTC")


def test_fred_audit_rejects_realtime_end_before_start():
    data = vintage_frame()
    data.loc[0, "realtime_end"] = "2023-12-30"
    with pytest.raises(ValueError, match="realtime_end cannot precede realtime_start"):
        audit_fred_point_in_time(data)


def test_fred_audit_rejects_duplicate_vintage_key():
    data = pd.concat([vintage_frame(), vintage_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate FRED vintage keys"):
        audit_fred_point_in_time(data)
