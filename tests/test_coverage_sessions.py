from datetime import date

import pandas as pd

from market_predictor.historical_coverage import DATASET_END, DATASET_START, coverage_plan, required_sources
from market_predictor.session_calendar import audit_market_timestamps, is_market_session, session_close


def test_coverage_is_complete_year_and_required_sources_are_declared():
    assert DATASET_START == date(2000, 1, 3)
    assert DATASET_END == date(2025, 12, 31)
    required = {source.source_id for source in required_sources()}
    assert "stooq_spx_daily" in required
    assert "fred_vintages" in required
    assert "gdelt_events" in required

    plan = coverage_plan()
    assert len(plan) == 8
    assert {source.dataset for source in plan if source.source_id == "fred_vintages"} == {
        "FEDFUNDS", "DGS10", "CPIAUCSL", "UNRATE", "VIXCLS"
    }
    assert all(source.point_in_time for source in plan if source.source_id == "fred_vintages")


def test_session_calendar_handles_weekend_holiday_and_dst_close():
    assert not is_market_session(date(2025, 1, 1))
    assert not is_market_session(date(2025, 1, 4))
    assert is_market_session(date(2025, 1, 2))
    # 16:00 ET = 21:00 UTC in winter and 20:00 UTC in summer.
    assert session_close(date(2025, 1, 2)).hour == 21
    assert session_close(date(2025, 7, 1)).hour == 20


def test_timestamp_audit_flags_after_close_and_weekend():
    index = pd.DatetimeIndex([
        "2025-01-02T20:59:00Z",
        "2025-01-02T21:01:00Z",
        "2025-01-04T15:00:00Z",
    ])
    audit = audit_market_timestamps(index)
    assert bool(audit.iloc[0].at_or_before_close)
    assert bool(audit.iloc[1].after_close)
    assert bool(audit.iloc[2].weekend)
