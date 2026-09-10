from datetime import datetime, timezone

import pytest

from market_predictor.decision_engine import DecisionConfig, PredictionSnapshot, decide


def snapshot(**overrides):
    values = dict(
        timestamp=datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
        symbol="SPX",
        model_version="model-c",
        feature_schema_version="features-v1",
        experiment_id="exp-c-lockbox",
        probability=0.70,
        confidence=0.80,
        expected_edge_bps=20.0,
        estimated_cost_bps=5.0,
        slippage_bps=2.0,
        signal_age_seconds=0,
        data_status="OK",
        model_approved=True,
        calibration_available=True,
        lockbox_released=True,
        regime_valid=True,
        current_exposure=0.0,
        proposed_exposure=0.10,
        proposed_turnover=0.10,
        drawdown=0.01,
    )
    values.update(overrides)
    return PredictionSnapshot(**values)


def test_valid_snapshot_produces_long_and_serializes():
    decision = decide(snapshot())
    assert decision.action == "LONG"
    assert decision.reason_codes == ("LONG_THRESHOLD",)
    assert decision.edge_after_costs_bps == pytest.approx(12.0)
    assert decision.to_dict()["contract_version"] == "decision-v1"
    assert '"action":"LONG"' in decision.to_json()


@pytest.mark.parametrize(
    "override,reason",
    [
        ({"data_status": "STALE"}, "DATA_STALE"),
        ({"signal_age_seconds": 86_401}, "DATA_STALE"),
        ({"model_approved": False}, "MODEL_NOT_APPROVED"),
        ({"calibration_available": False}, "CALIBRATION_UNAVAILABLE"),
        ({"lockbox_released": False}, "LOCKBOX_NOT_RELEASED"),
        ({"regime_valid": False}, "REGIME_OUTSIDE_VALIDATED"),
        ({"confidence": 0.54}, "CONFIDENCE_BELOW_MIN"),
        ({"expected_edge_bps": 8.0}, "EDGE_AFTER_COSTS_NON_POSITIVE"),
        ({"proposed_exposure": 1.01}, "EXPOSURE_LIMIT"),
        ({"proposed_turnover": 0.26}, "TURNOVER_LIMIT"),
        ({"drawdown": 0.101}, "DRAWDOWN_LIMIT"),
    ],
)
def test_any_failed_gate_forces_no_trade(override, reason):
    decision = decide(snapshot(**override))
    assert decision.action == "NO_TRADE"
    assert reason in decision.reason_codes


def test_failed_gate_wins_over_high_probability():
    decision = decide(snapshot(probability=0.99, model_approved=False))
    assert decision.action == "NO_TRADE"
    assert "MODEL_NOT_APPROVED" in decision.reason_codes
    assert "LONG_THRESHOLD" not in decision.reason_codes


def test_middle_probability_is_flat_when_all_gates_pass():
    decision = decide(snapshot(probability=0.50))
    assert decision.action == "FLAT"
    assert decision.reason_codes == ("NO_DIRECTIONAL_EDGE",)


def test_short_threshold_is_explicit():
    decision = decide(snapshot(probability=0.40))
    assert decision.action == "SHORT"
    assert decision.reason_codes == ("SHORT_THRESHOLD",)


def test_threshold_configuration_is_validated():
    with pytest.raises(ValueError):
        DecisionConfig(long_threshold=0.49)
    with pytest.raises(ValueError):
        DecisionConfig(short_threshold=0.51)
    with pytest.raises(ValueError):
        DecisionConfig(long_threshold=0.55, short_threshold=0.55)


def test_snapshot_requires_timezone_aware_timestamp():
    with pytest.raises(ValueError):
        snapshot(timestamp=datetime(2026, 1, 2, 12))
