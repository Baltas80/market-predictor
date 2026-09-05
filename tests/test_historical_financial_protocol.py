from datetime import date

import pandas as pd
import pytest

from market_predictor.experiment_protocol import SharedExperimentProtocol, assert_common_protocol, assert_oos_configuration_frozen
from market_predictor.financial_matrix import evaluate_financial_matrix, period_stability
from market_predictor.research_schema import (
    EVENT_COLUMNS,
    deduplicate_events,
    normalize_event_sources,
    validate_event_frame,
    validate_market_frame,
)


def _protocol(index_hash="idx"):
    return SharedExperimentProtocol(
        lockbox_start=date(2025, 1, 1), lockbox_end=date(2025, 12, 31),
        purge_gap=5, horizon=5, transaction_cost_bps=5, slippage_bps=2,
        observations=20, prediction_index_hash=index_hash,
        dataset_hash="data", code_version="code",
    )


def test_shared_protocol_rejects_any_abc_mismatch():
    assert_common_protocol({"A": _protocol(), "B": _protocol(), "C": _protocol()})
    with pytest.raises(ValueError):
        assert_common_protocol({"A": _protocol(), "B": _protocol("other")})


def test_oos_configuration_must_match_recorded_fingerprint():
    assert_oos_configuration_frozen(configuration_fingerprint="abc", recorded_fingerprint="abc")
    with pytest.raises(ValueError):
        assert_oos_configuration_frozen(configuration_fingerprint="abc", recorded_fingerprint="changed")


def test_event_schema_separates_event_publication_and_availability():
    frame = pd.DataFrame({
        "event_id": ["e1", "e2"],
        "event_time": ["2025-01-01T09:00:00Z", "2025-01-02T09:00:00Z"],
        "published_at": ["2025-01-01T10:00:00Z", "2025-01-02T10:00:00Z"],
        "available_at": ["2025-01-01T10:05:00Z", "2025-01-02T10:05:00Z"],
        "source_id": ["src", "src"], "category": ["war_conflict", "sanctions"],
        "severity": [0.8, 0.4], "country": ["US", "US"], "entity": ["x", "y"],
        "sector": [None, None], "duration_days": [0, 0],
        "media_intensity": [1, 2], "surprise": [0.2, 0.1],
    })
    validate_event_frame(frame)
    admitted = frame.loc[frame.available_at <= "2025-01-01T10:06:00Z"]
    assert admitted.event_id.tolist() == ["e1"]


def test_event_dedup_and_source_normalization_are_deterministic():
    raw = pd.DataFrame({
        "event_id": ["e1", "e1"], "sql_date": ["2025-01-01", "2025-01-01"],
        "published_at": ["2025-01-01T12:00:00Z", "2025-01-01T12:00:00Z"],
        "available_at": ["2025-01-01T12:02:00Z", "2025-01-01T12:05:00Z"],
        "category": ["war_conflict", "war_conflict"], "severity": [0.5, 0.5],
    })
    normalized = normalize_event_sources(raw, source_id="gdelt")
    result = deduplicate_events(normalized)
    assert result.event_id.tolist() == ["e1"]
    assert result.available_at.iloc[0] == pd.Timestamp("2025-01-01T12:02:00Z")
    assert set(result.columns) == set(EVENT_COLUMNS)


def test_financial_matrix_has_fixed_3x2_grid_plus_random_baseline():
    idx = pd.date_range("2025-01-01", periods=30, freq="D", tz="UTC")
    close = pd.Series(range(100, 130), index=idx, dtype=float)
    predictions = {name: pd.DataFrame({"prob_up": 0.6, "close": close}, index=idx) for name in ("A", "B", "C")}
    result = evaluate_financial_matrix(predictions, benchmark=pd.DataFrame({"close": close}, index=idx))
    assert len(result) == 4 * 3 * 2
    assert set(result.experiment) == {"A", "B", "C", "random"}
    assert result[["total_return", "max_drawdown", "sharpe", "sortino", "turnover"]].notna().all().all()


def test_financial_matrix_rejects_mismatched_prediction_indices():
    idx = pd.date_range("2025-01-01", periods=5, tz="UTC")
    other = pd.date_range("2025-01-02", periods=5, tz="UTC")
    close = pd.Series(range(100, 105), index=idx, dtype=float)
    frame = pd.DataFrame({"prob_up": 0.6, "close": close}, index=idx)
    bad = pd.DataFrame({"prob_up": 0.6, "close": range(100, 105)}, index=other)
    with pytest.raises(ValueError):
        evaluate_financial_matrix({"A": frame, "B": bad}, benchmark=pd.DataFrame({"close": close}, index=idx))


def test_period_stability_reports_turnover_and_risk_metrics():
    idx = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")
    backtest = pd.DataFrame({
        "strategy_return": [0.01, -0.005] * 5,
        "position_change": [1, 0] * 5,
    }, index=idx)
    result = period_stability(backtest, periods={"first": ("2025-01-01", "2025-01-05"), "second": ("2025-01-06", "2025-01-10")})
    assert result.period.tolist() == ["first", "second"]
    assert (result.observations == 5).all()
    assert (result.turnover == 3).all()


def test_market_schema_requires_timezone_and_finite_values():
    idx = pd.date_range("2025-01-01", periods=2, tz="UTC")
    frame = pd.DataFrame({"open": [1, 2], "high": [2, 3], "low": [1, 2], "close": [2, 3], "volume": [10, 20]}, index=idx)
    validate_market_frame(frame)
    bad = frame.copy(); bad.iloc[0, 3] = float("inf")
    with pytest.raises(ValueError):
        validate_market_frame(bad)
