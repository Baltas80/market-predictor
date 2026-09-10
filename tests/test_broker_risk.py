from market_predictor.brokers import AccountSnapshot, OrderRequest
from market_predictor.brokers.risk import RiskLimits, check_order


def account() -> AccountSnapshot:
    return AccountSnapshot(currency="USD", cash=10_000, equity=10_000, buying_power=10_000)


def test_kill_switch_blocks_order():
    decision = check_order(
        OrderRequest(symbol="SPY", side="buy", quantity=1),
        reference_price=100,
        account=account(),
        limits=RiskLimits(),
        kill_switch=True,
    )
    assert not decision.allowed
    assert "kill switch" in decision.reason


def test_order_notional_limit_blocks_order():
    decision = check_order(
        OrderRequest(symbol="SPY", side="buy", quantity=6),
        reference_price=100,
        account=account(),
        limits=RiskLimits(max_order_notional=500),
    )
    assert not decision.allowed
    assert "notional" in decision.reason


def test_valid_order_passes():
    decision = check_order(
        OrderRequest(symbol="SPY", side="buy", quantity=2),
        reference_price=100,
        account=account(),
        limits=RiskLimits(),
    )
    assert decision.allowed
