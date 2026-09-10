from market_predictor.brokers import AccountSnapshot, OrderRequest, Position
from market_predictor.brokers.risk import RiskLimits, check_order


def account(*positions: Position) -> AccountSnapshot:
    return AccountSnapshot(
        currency="USD",
        cash=10_000,
        equity=10_000,
        buying_power=10_000,
        positions=positions,
    )


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


def test_resulting_position_limit_blocks_order():
    decision = check_order(
        OrderRequest(symbol="SPY", side="buy", quantity=2),
        reference_price=100,
        account=account(Position("SPY", 19)),
        limits=RiskLimits(max_order_notional=500, max_position_notional=2_000),
    )
    assert not decision.allowed
    assert "position" in decision.reason


def test_sell_can_reduce_position_under_limit():
    decision = check_order(
        OrderRequest(symbol="SPY", side="sell", quantity=2),
        reference_price=100,
        account=account(Position("SPY", 19)),
        limits=RiskLimits(max_order_notional=500, max_position_notional=2_000),
    )
    assert decision.allowed


def test_short_exposure_is_also_limited():
    decision = check_order(
        OrderRequest(symbol="SPY", side="sell", quantity=2),
        reference_price=100,
        account=account(Position("SPY", -19)),
        limits=RiskLimits(max_order_notional=500, max_position_notional=2_000),
    )
    assert not decision.allowed
    assert "position" in decision.reason


def test_valid_order_passes():
    decision = check_order(
        OrderRequest(symbol="SPY", side="buy", quantity=2),
        reference_price=100,
        account=account(),
        limits=RiskLimits(),
    )
    assert decision.allowed
