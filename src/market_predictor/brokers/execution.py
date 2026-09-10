"""Broker-neutral execution engine with risk isolation and idempotency."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from .base import BrokerAdapter, OrderRequest, OrderResult
from .risk import RiskDecision, RiskLimits, check_order


@dataclass(frozen=True)
class ExecutionDecision:
    """Auditable outcome of an execution attempt."""

    client_order_id: str
    status: str
    risk: RiskDecision
    order: OrderRequest
    result: OrderResult | None = None
    decided_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ExecutionEngine:
    """Route approved signals to a broker adapter without bypassing the risk gate.

    The engine is deliberately broker-neutral. It keeps an in-memory idempotency
    ledger for a single process; production adapters should additionally enforce
    idempotency at the broker/API boundary.
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        *,
        limits: RiskLimits | None = None,
        daily_loss_provider: Callable[[], float] | None = None,
        kill_switch_provider: Callable[[], bool] | None = None,
    ) -> None:
        self.broker = broker
        self.limits = limits or RiskLimits()
        self.daily_loss_provider = daily_loss_provider or (lambda: 0.0)
        self.kill_switch_provider = kill_switch_provider or (lambda: False)
        self._ledger: dict[str, ExecutionDecision] = {}

    @staticmethod
    def _with_client_order_id(order: OrderRequest, client_order_id: str) -> OrderRequest:
        if order.client_order_id == client_order_id:
            return order
        return OrderRequest(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            limit_price=order.limit_price,
            client_order_id=client_order_id,
        )

    def execute(
        self,
        order: OrderRequest,
        *,
        reference_price: float,
        client_order_id: str,
    ) -> ExecutionDecision:
        """Risk-check and submit an order exactly once per client order id.

        A reused client ID is idempotent only when it refers to the same
        normalized order payload. Reusing an ID for a different order raises
        instead of returning an unrelated previous result.
        """
        if not client_order_id.strip():
            raise ValueError("client_order_id must not be empty")

        order = self._with_client_order_id(order, client_order_id)
        previous = self._ledger.get(client_order_id)
        if previous is not None:
            if previous.order != order:
                raise ValueError("client_order_id already belongs to a different order")
            return previous

        account = self.broker.get_account()
        risk = check_order(
            order,
            reference_price=reference_price,
            account=account,
            limits=self.limits,
            daily_loss=self.daily_loss_provider(),
            kill_switch=self.kill_switch_provider(),
        )
        decided_at = datetime.now(timezone.utc)
        if not risk.allowed:
            decision = ExecutionDecision(
                client_order_id=client_order_id,
                status="rejected_by_risk",
                risk=risk,
                order=order,
                decided_at=decided_at,
            )
            self._ledger[client_order_id] = decision
            return decision

        result = self.broker.submit_order(order)
        decision = ExecutionDecision(
            client_order_id=client_order_id,
            status=result.status,
            risk=risk,
            order=order,
            result=result,
            decided_at=decided_at,
        )
        self._ledger[client_order_id] = decision
        return decision

    def get_decision(self, client_order_id: str) -> ExecutionDecision | None:
        """Return a previously recorded decision, if any."""
        return self._ledger.get(client_order_id)

    def audit_log(self) -> tuple[ExecutionDecision, ...]:
        """Return decisions in deterministic insertion order for auditing."""
        return tuple(self._ledger.values())
