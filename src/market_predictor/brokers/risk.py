"""Execution-time risk gate kept outside the predictive model."""

from __future__ import annotations

from dataclasses import dataclass

from .base import AccountSnapshot, OrderRequest


@dataclass(frozen=True)
class RiskLimits:
    max_order_notional: float = 500.0
    max_position_notional: float = 2_000.0
    max_daily_loss: float = 200.0

    def __post_init__(self) -> None:
        if self.max_order_notional <= 0 or self.max_position_notional <= 0 or self.max_daily_loss <= 0:
            raise ValueError("risk limits must be positive")


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str


def check_order(
    order: OrderRequest,
    *,
    reference_price: float,
    account: AccountSnapshot,
    limits: RiskLimits,
    daily_loss: float = 0.0,
    kill_switch: bool = False,
) -> RiskDecision:
    """Validate an order without allowing the predictive model to bypass limits."""
    if kill_switch:
        return RiskDecision(False, "kill switch is active")
    if reference_price <= 0:
        return RiskDecision(False, "reference price must be positive")
    if daily_loss >= limits.max_daily_loss:
        return RiskDecision(False, "daily loss limit reached")

    notional = order.quantity * reference_price
    if notional > limits.max_order_notional:
        return RiskDecision(False, "order notional exceeds limit")
    if notional > account.buying_power:
        return RiskDecision(False, "insufficient buying power")

    current_quantity = sum(
        position.quantity for position in account.positions if position.symbol == order.symbol
    )
    signed_quantity = order.quantity if order.side == "buy" else -order.quantity
    projected_position_notional = abs(current_quantity + signed_quantity) * reference_price
    if projected_position_notional > limits.max_position_notional:
        return RiskDecision(False, "projected position notional exceeds limit")

    return RiskDecision(True, "order passed risk checks")
