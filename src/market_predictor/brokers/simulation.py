"""Deterministic broker simulator for execution-engine tests.

This adapter never connects to a network or external account. It exists to
exercise order validation and state transitions before a real broker adapter
is introduced.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .base import AccountSnapshot, BrokerAdapter, OrderRequest, OrderResult, Position


class SimulatedBroker(BrokerAdapter):
    def __init__(self, *, cash: float = 100_000.0) -> None:
        if cash < 0:
            raise ValueError("cash must not be negative")
        self._cash = cash
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, OrderResult] = {}
        self._next_order_id = 1

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            currency="USD",
            cash=self._cash,
            equity=self._cash,
            buying_power=self._cash,
            positions=self.get_positions(),
        )

    def get_positions(self) -> tuple[Position, ...]:
        return tuple(self._positions.values())

    def submit_order(self, order: OrderRequest) -> OrderResult:
        if order.order_type != "market":
            return self._record(order, "rejected", message="simulation only supports market orders")

        # A simulator deliberately requires an explicit reference price rather
        # than inventing a market price. Real adapters receive execution data
        # from the broker and report the actual fill back to the engine.
        return self._record(order, "rejected", message="simulation requires an execution price")

    def cancel_order(self, broker_order_id: str) -> OrderResult:
        existing = self._orders.get(broker_order_id)
        if existing is None:
            raise KeyError(broker_order_id)
        if existing.status in {"filled", "cancelled", "rejected"}:
            return existing
        cancelled = OrderResult(
            broker_order_id=existing.broker_order_id,
            status="cancelled",
            submitted_at=existing.submitted_at,
            message="cancelled by simulation",
        )
        self._orders[broker_order_id] = cancelled
        return cancelled

    def _record(self, order: OrderRequest, status: str, *, message: str) -> OrderResult:
        broker_order_id = f"SIM-{self._next_order_id:08d}"
        self._next_order_id += 1
        result = OrderResult(
            broker_order_id=broker_order_id,
            status=status,  # type: ignore[arg-type]
            submitted_at=datetime.now(timezone.utc),
            message=message,
        )
        self._orders[broker_order_id] = result
        return result
