"""Broker execution interfaces for Market Predictor.

This package deliberately starts with a broker-agnostic interface. Live broker
credentials and live order routing are not part of the research pipeline.
"""

from .base import AccountSnapshot, BrokerAdapter, OrderRequest, OrderResult, Position
from .execution import ExecutionDecision, ExecutionEngine
from .risk import RiskDecision, RiskLimits

__all__ = [
    "AccountSnapshot",
    "BrokerAdapter",
    "ExecutionDecision",
    "ExecutionEngine",
    "OrderRequest",
    "OrderResult",
    "Position",
    "RiskDecision",
    "RiskLimits",
]
