from __future__ import annotations

import pandas as pd
import pytest

from market_predictor.time_contract import (
    ObservationWindow,
    assert_point_in_time,
    next_session_decision_time,
)


def test_observation_window_requires_available_before_decision_and_future_target() -> None:
    window = ObservationWindow(
        available_at=pd.Timestamp("2020-01-01T12:00:00Z"),
        decision_time=pd.Timestamp("2020-01-02T00:00:00Z"),
        target_end=pd.Timestamp("2020-01-07T00:00:00Z"),
    )
    window.validate()

    with pytest.raises(ValueError):
        ObservationWindow(
            pd.Timestamp("2020-01-03T00:00:00Z"),
            pd.Timestamp("2020-01-02T00:00:00Z"),
            pd.Timestamp("2020-01-07T00:00:00Z"),
        ).validate()


def test_series_contract_rejects_future_information() -> None:
    with pytest.raises(ValueError, match="Point-in-time"):
        assert_point_in_time(
            pd.Series(["2020-01-03T00:00:00Z"]),
            pd.Series(["2020-01-02T00:00:00Z"]),
            pd.Series(["2020-01-07T00:00:00Z"]),
        )


def test_next_session_is_strictly_after_availability() -> None:
    market = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    assert next_session_decision_time(market, pd.Timestamp("2020-01-01T12:00:00Z")) == market[1]
    assert next_session_decision_time(market, pd.Timestamp("2019-12-31T23:00:00Z")) == market[0]


def test_naive_timestamp_is_rejected_by_window() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ObservationWindow(
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-02", tz="UTC"),
            pd.Timestamp("2020-01-07", tz="UTC"),
        ).validate()
