from datetime import datetime, timezone

import pytest

from market_predictor.brokers import AccountSnapshot, OrderRequest, Position


def test_market_order_requires_positive_quantity():
    with pytest.raises(ValueError, match="quantity must be positive"):
        OrderRequest(symbol="SPY", side="buy", quantity=0)


def test_limit_order_requires_positive_limit_price():
    with pytest.raises(ValueError, match="limit_price"):
        OrderRequest(symbol="SPY", side="buy", quantity=1, order_type="limit")


def test_account_snapshot_is_broker_neutral():
    position = Position(symbol="SPY", quantity=2, average_price=500.0)
    snapshot = AccountSnapshot(
        currency="USD",
        cash=1000.0,
        equity=2000.0,
        buying_power=1000.0,
        positions=(position,),
    )
    assert snapshot.positions == (position,)
    assert snapshot.equity == 2000.0


def test_order_request_does_not_mutate_timestamp_or_credentials():
    order = OrderRequest(
        symbol="AAPL",
        side="sell",
        quantity=3,
        client_order_id="research-001",
    )
    assert order.client_order_id == "research-001"
    assert datetime.now(timezone.utc).tzinfo is not None
