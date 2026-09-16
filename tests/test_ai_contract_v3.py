from datetime import datetime, timedelta, timezone

import pytest

from market_predictor.ai_contract_v3 import AIEventSignalV3, assert_ai_available, validate_ai_feature_names


def signal(at):
    return AIEventSignalV3("e1", "src1", "POLITICAL_CRISIS", .5, .4, .8, .9, at)


def test_ai_signal_must_be_available_before_decision():
    now = datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert_ai_available(signal(now - timedelta(minutes=1)), now)
    with pytest.raises(ValueError):
        assert_ai_available(signal(now + timedelta(minutes=1)), now)


def test_ai_features_reject_target_like_names():
    with pytest.raises(ValueError):
        validate_ai_feature_names(["ai_severity", "future_return"])


def test_ai_signal_features_are_auditable():
    now = datetime(2026, 1, 2, tzinfo=timezone.utc)
    features = signal(now).as_features()
    assert features["available_at"] == now
