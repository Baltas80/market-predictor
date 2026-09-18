from datetime import datetime, timezone

import pytest

from market_predictor.brokers.base import AccountSnapshot, OrderRequest, OrderResult, Position
from market_predictor.brokers.execution import ExecutionEngine
from market_predictor.brokers.risk import RiskLimits


class FakeBroker:
    def __init__(self, positions=()):
        self.submissions = []
        self.positions = tuple(positions)

    def get_account(self):
        return AccountSnapshot("USD", 10000, 10000, 5000, self.positions)

    def get_positions(self):
        return self.positions

    def submit_order(self, order):
        self.submissions.append(order)
        return OrderResult("broker-1", "accepted", datetime.now(timezone.utc), order.quantity, 10.0)

    def cancel_order(self, broker_order_id):
        raise NotImplementedError


def test_execution_passes_risk_and_submits_once():
    broker = FakeBroker()
    engine = ExecutionEngine(broker, limits=RiskLimits(max_order_notional=200))
    order = OrderRequest("ABC", "buy", 10)

    first = engine.execute(order, reference_price=10, client_order_id="sig-1")
    second = engine.execute(order, reference_price=10, client_order_id="sig-1")

    assert first.status == "accepted"
    assert first.risk.allowed is True
    assert second is first
    assert len(broker.submissions) == 1
    assert broker.submissions[0].client_order_id == "sig-1"
    assert engine.audit_log() == (first,)


def test_execution_rejects_reuse_of_client_id_for_different_order():
    broker = FakeBroker()
    engine = ExecutionEngine(broker)
    engine.execute(OrderRequest("ABC", "buy", 1), reference_price=10, client_order_id="sig-1")

    with pytest.raises(ValueError, match="different order"):
        engine.execute(OrderRequest("ABC", "buy", 2), reference_price=10, client_order_id="sig-1")

    assert len(broker.submissions) == 1


def test_execution_never_submits_when_risk_rejects():
    broker = FakeBroker()
    engine = ExecutionEngine(broker, limits=RiskLimits(max_order_notional=50))
    decision = engine.execute(OrderRequest("ABC", "buy", 10), reference_price=10, client_order_id="sig-2")

    assert decision.status == "rejected_by_risk"
    assert decision.risk.allowed is False
    assert broker.submissions == []


def test_kill_switch_is_independent_of_order_signal():
    broker = FakeBroker()
    engine = ExecutionEngine(broker, kill_switch_provider=lambda: True)
    decision = engine.execute(OrderRequest("ABC", "buy", 1), reference_price=10, client_order_id="sig-3")

    assert decision.status == "rejected_by_risk"
    assert decision.risk.reason == "kill switch is active"
    assert broker.submissions == []
