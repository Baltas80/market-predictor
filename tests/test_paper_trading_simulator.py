from datetime import datetime, timedelta, timezone

import pytest

from market_predictor.paper_trading import (
    PaperSignal,
    PaperTradingSimulator,
    simulate_paper_predictions,
)


UTC = timezone.utc


def signal(n: int, position: int) -> PaperSignal:
    return PaperSignal(
        signal_id=f"s-{n}",
        decision_time=datetime(2026, 1, n, tzinfo=UTC),
        probability_up=0.7 if position == 1 else 0.3,
        confidence=0.8,
        position=position,
        expected_risk=0.02,
    )


def test_virtual_account_grows_on_profitable_long_interval():
    sim = PaperTradingSimulator(initial_cash=100_000, allocation=1.0)
    sim.step(signal(1, 1), 100.0)
    account = sim.close(datetime(2026, 1, 3, tzinfo=UTC), 110.0)

    assert account.equity == pytest.approx(110_000.0)
    frame = sim.experience_frame()
    assert len(frame) == 1
    assert frame.loc[0, "realized_return"] == pytest.approx(0.10)
    assert frame.loc[0, "pnl"] == pytest.approx(10_000.0)


def test_learning_outcome_is_strictly_after_decision():
    sim = PaperTradingSimulator()
    sim.step(signal(1, 1), 100.0)
    with pytest.raises(ValueError, match="follow decision_time"):
        sim.close(datetime(2026, 1, 1, tzinfo=UTC), 101.0)


def test_duplicate_signal_is_rejected():
    sim = PaperTradingSimulator()
    first = signal(1, 1)
    sim.step(first, 100.0)
    with pytest.raises(ValueError, match="signal_id already exists"):
        sim.step(first, 101.0)


def test_chronology_is_enforced():
    sim = PaperTradingSimulator()
    sim.step(signal(2, 1), 100.0)
    earlier = PaperSignal(
        signal_id="earlier",
        decision_time=datetime(2026, 1, 1, tzinfo=UTC),
        probability_up=0.5,
        confidence=0.5,
        position=0,
        expected_risk=0.0,
    )
    with pytest.raises(ValueError, match="strictly chronological"):
        sim.step(earlier, 101.0)


def test_transaction_costs_and_slippage_reduce_realized_return():
    sim = PaperTradingSimulator(
        initial_cash=100_000,
        transaction_cost_bps=10,
        slippage_bps=5,
    )
    sim.step(signal(1, 1), 100.0)
    sim.close(datetime(2026, 1, 2, tzinfo=UTC), 110.0)
    realized = sim.experience_frame().loc[0, "realized_return"]
    assert realized < 0.10


def test_simulate_requires_one_price_per_signal():
    with pytest.raises(ValueError, match="equal length"):
        simulate_paper_predictions([signal(1, 1)], [])


def test_experience_frame_has_expected_audit_columns():
    sim = PaperTradingSimulator()
    sim.step(signal(1, 0), 100.0)
    sim.step(signal(2, 0), 101.0)
    frame = sim.experience_frame()
    assert list(frame.columns) == [
        "signal_id",
        "decision_time",
        "outcome_time",
        "probability_up",
        "confidence",
        "position",
        "entry_price",
        "exit_price",
        "notional",
        "realized_return",
        "pnl",
        "reward",
    ]
    assert frame.loc[0, "outcome_time"] > frame.loc[0, "decision_time"]
