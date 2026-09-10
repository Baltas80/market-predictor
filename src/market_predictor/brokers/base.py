"""Broker-neutral execution contracts.

The research engine should depend on these contracts rather than on a specific
broker SDK. Implementations must enforce their own authentication, idempotency,
and risk controls before sending live orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]
OrderStatus = Literal["accepted", "filled", "rejected", "cancelled"]


@dataclass(frozen=True)
class OrderRequest:
    """An explicit order proposal produced after risk checks."""

    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = "market"
    limit_price: float | None = None
    client_order_id: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol must not be empty")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type == "limit" and (self.limit_price is None or self.limit_price <= 0):
            raise ValueError("limit orders require a positive limit_price")


@dataclass(frozen=True)
class OrderResult:
    """Broker acknowledgement for an order submission."""

    broker_order_id: str
    status: OrderStatus
    submitted_at: datetime
    filled_quantity: float = 0.0
    average_fill_price: float | None = None
    message: str = ""


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    average_price: float | None = None


@dataclass(frozen=True)
class AccountSnapshot:
    currency: str
    cash: float
    equity: float
    buying_power: float
    positions: tuple[Position, ...] = ()


class BrokerAdapter(Protocol):
    """Minimal interface required by an execution engine."""

    def get_account(self) -> AccountSnapshot: ...

    def get_positions(self) -> tuple[Position, ...]: ...

    def submit_order(self, order: OrderRequest) -> OrderResult: ...

    def cancel_order(self, broker_order_id: str) -> OrderResult: ...
