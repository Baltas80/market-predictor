from __future__ import annotations

import pandas as pd

from market_predictor.data_quality import quality_report


def test_quality_report_accepts_valid_inputs() -> None:
    market = pd.DataFrame(
        {"open": [10.0], "high": [12.0], "low": [9.0], "close": [11.0], "volume": [100.0]},
        index=pd.DatetimeIndex(["2020-01-01"]),
    )
    macro = pd.DataFrame({
        "date": ["2020-01-01"], "value": [1.0],
        "realtime_start": ["2020-01-02"], "realtime_end": ["2020-01-03"],
    })
    events = pd.DataFrame({
        "event_id": ["e1"], "published_at": ["2020-01-01T12:00:00Z"], "severity": [0.8],
        "duration_days": [2], "media_intensity": [3],
    })
    assert quality_report(market=market, macro=macro, events=events)["passed"]


def test_quality_report_rejects_impossible_market_ohlc() -> None:
    market = pd.DataFrame(
        {"open": [15.0], "high": [12.0], "low": [9.0], "close": [11.0], "volume": [100.0]},
        index=pd.DatetimeIndex(["2020-01-01"]),
    )
    report = quality_report(market=market)
    assert not report["passed"]
    assert any(c["name"] == "market_ohlc_bounds" and not c["passed"] for c in report["checks"])


def test_quality_report_rejects_duplicate_macro_vintage() -> None:
    macro = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-01"], "value": [1.0, 1.1],
        "realtime_start": ["2020-01-02", "2020-01-02"],
        "realtime_end": ["2020-01-03", "2020-01-03"],
    })
    report = quality_report(macro=macro)
    assert not report["passed"]


def test_quality_report_rejects_event_severity_outside_range() -> None:
    events = pd.DataFrame({
        "event_id": ["e1"], "published_at": ["2020-01-01T12:00:00Z"], "severity": [1.5],
    })
    report = quality_report(events=events)
    assert not report["passed"]
