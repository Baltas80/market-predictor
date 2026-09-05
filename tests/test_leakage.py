from __future__ import annotations

import pandas as pd
import pytest

from market_predictor.leakage import (
    assert_event_availability_before_decision,
    assert_monotonic_unique_index,
    assert_target_is_future,
    audit_feature_columns,
)


def test_index_must_be_unique_and_chronological() -> None:
    assert_monotonic_unique_index(pd.date_range("2020-01-01", periods=3, freq="D"))
    with pytest.raises(ValueError):
        assert_monotonic_unique_index(pd.DatetimeIndex(["2020-01-02", "2020-01-01"]))


def test_target_must_end_after_decision() -> None:
    valid = pd.DataFrame({
        "decision_time": ["2020-01-01"],
        "target_end": ["2020-01-06"],
    })
    assert_target_is_future(valid)
    invalid = pd.DataFrame({
        "decision_time": ["2020-01-06"],
        "target_end": ["2020-01-06"],
    })
    with pytest.raises(ValueError):
        assert_target_is_future(invalid)


def test_event_availability_cannot_exceed_latest_decision() -> None:
    events = pd.DataFrame({"published_at": ["2020-01-01T12:00:00Z"]})
    decisions = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    assert_event_availability_before_decision(events, decision_times=decisions)
    late = pd.DataFrame({"published_at": ["2020-01-04T12:00:00Z"]})
    with pytest.raises(ValueError):
        assert_event_availability_before_decision(late, decision_times=decisions)


def test_feature_name_audit_flags_common_future_tokens() -> None:
    suspicious = audit_feature_columns(["return_1d", "future_return", "rolling_vol", "target"])
    assert suspicious == ["future_return", "target"]
