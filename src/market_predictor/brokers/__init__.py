"""Broker execution interfaces for Market Predictor.

This package deliberately starts with a broker-agnostic interface. Live broker
credentials and live order routing are not part of the research pipeline.
"""

from .base import AccountSnapshot, BrokerAdapter, OrderRequest, OrderResult, Position

__all__ = [
    "AccountSnapshot",
    "BrokerAdapter",
    "OrderRequest",
    "OrderResult",
    "Position",
]
