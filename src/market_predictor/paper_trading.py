"""Deterministic virtual trading and learning experiences.

The module is deliberately broker-free. It can execute model signals against
explicitly supplied historical prices, but it has no live-order or credential path.
Resolved virtual trades form an auditable experience table for later training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class PaperSignal:
    """Validated model signal; this remains compatible with the audit contract."""

    signal_id: str
    decision_time: datetime
    probability_up: float
    confidence: float
    position: int
    expected_risk: float

    def validate(self) -> None:
        if not self.signal_id:
            raise ValueError("signal_id is required")
        if self.decision_time.tzinfo is None:
            raise ValueError("decision_time must be timezone-aware")
        if not 0 <= self.probability_up <= 1:
            raise ValueError("probability_up must be in [0, 1]")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if self.position not in (-1, 0, 1):
            raise ValueError("position must be -1, 0 or 1")
        if self.expected_risk < 0:
            raise ValueError("expected_risk cannot be negative")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class PaperTrade:
    """One resolved virtual holding interval."""

    signal_id: str
    decision_time: datetime
    outcome_time: datetime
    probability_up: float
    confidence: float
    position: int
    entry_price: float
    exit_price: float
    notional: float
    realized_return: float
    pnl: float
    reward: float


@dataclass(frozen=True)
class PaperAccount:
    """Mark-to-market state of the virtual account."""

    initial_cash: float
    cash: float
    equity: float
    position_units: float
    mark_price: float


class PaperTradingSimulator:
    """Long/short/flat virtual account with deterministic execution costs."""

    def __init__(
        self,
        *,
        initial_cash: float = 100_000.0,
        allocation: float = 1.0,
        transaction_cost_bps: float = 0.0,
        slippage_bps: float = 0.0,
    ) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not 0 < allocation <= 1:
            raise ValueError("allocation must be in (0,1]")
        if transaction_cost_bps < 0 or slippage_bps < 0:
            raise ValueError("costs cannot be negative")
        self.initial_cash = float(initial_cash)
        self.allocation = float(allocation)
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.slippage_bps = float(slippage_bps)
        self._cash = self.initial_cash
        self._units = 0.0
        self._active_signal: PaperSignal | None = None
        self._entry_price: float | None = None
        self._entry_notional: float = 0.0
        self._trades: list[PaperTrade] = []
        self._seen_ids: set[str] = set()

    @property
    def trades(self) -> tuple[PaperTrade, ...]:
        return tuple(self._trades)

    def account(self, mark_price: float) -> PaperAccount:
        if mark_price <= 0:
            raise ValueError("mark_price must be positive")
        return PaperAccount(
            initial_cash=self.initial_cash,
            cash=self._cash,
            equity=self._cash + self._units * mark_price,
            position_units=self._units,
            mark_price=float(mark_price),
        )

    def _fill_price(self, price: float, delta_units: float) -> float:
        impact = self.slippage_bps / 10_000.0
        return price * (1.0 + impact if delta_units > 0 else 1.0 - impact)

    def _rebalance(self, target_units: float, price: float) -> None:
        delta = target_units - self._units
        if not delta:
            return
        fill = self._fill_price(price, delta)
        self._cash -= delta * fill
        self._cash -= abs(delta * fill) * self.transaction_cost_bps / 10_000.0
        self._units = target_units

    def step(self, signal: PaperSignal, execution_price: float) -> PaperAccount:
        """Apply a signal at its supplied decision-time price.

        A previous signal is resolved only when a later decision arrives. This
        makes the learning outcome strictly posterior to the decision timestamp.
        """
        signal.validate()
        if execution_price <= 0:
            raise ValueError("execution_price must be positive")
        if signal.signal_id in self._seen_ids:
            raise ValueError("signal_id already exists")
        if self._active_signal is not None and signal.decision_time <= self._active_signal.decision_time:
            raise ValueError("signals must be strictly chronological")

        if self._active_signal is not None:
            self._resolve_active(signal.decision_time, execution_price)

        equity = self.account(execution_price).equity
        target_units = signal.position * (equity * self.allocation) / execution_price
        self._rebalance(target_units, execution_price)
        self._active_signal = signal
        self._entry_price = execution_price
        self._entry_notional = abs(target_units * execution_price)
        self._seen_ids.add(signal.signal_id)
        return self.account(execution_price)

    def _resolve_active(self, outcome_time: datetime, exit_price: float) -> None:
        assert self._active_signal is not None
        assert self._entry_price is not None
        if outcome_time.tzinfo is None:
            raise ValueError("outcome_time must be timezone-aware")
        if outcome_time <= self._active_signal.decision_time:
            raise ValueError("outcome_time must follow decision_time")
        raw_return = (exit_price / self._entry_price - 1.0) * self._active_signal.position
        cost = 2.0 * (self.transaction_cost_bps + self.slippage_bps) / 10_000.0
        realized_return = raw_return - cost if self._active_signal.position else 0.0
        pnl = self._entry_notional * realized_return
        self._trades.append(
            PaperTrade(
                signal_id=self._active_signal.signal_id,
                decision_time=self._active_signal.decision_time,
                outcome_time=outcome_time,
                probability_up=self._active_signal.probability_up,
                confidence=self._active_signal.confidence,
                position=self._active_signal.position,
                entry_price=self._entry_price,
                exit_price=exit_price,
                notional=self._entry_notional,
                realized_return=realized_return,
                pnl=pnl,
                reward=realized_return,
            )
        )

    def close(self, outcome_time: datetime, exit_price: float) -> PaperAccount:
        """Close the active virtual position at an explicitly supplied later price."""
        if self._active_signal is None:
            raise ValueError("cannot close an empty simulator")
        self._resolve_active(outcome_time, exit_price)
        self._rebalance(0.0, exit_price)
        self._active_signal = None
        self._entry_price = None
        self._entry_notional = 0.0
        return self.account(exit_price)

    def experience_frame(self) -> pd.DataFrame:
        """Return resolved trades as an auditable training/diagnostic table."""
        columns = [
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
        if not self._trades:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([asdict(trade) for trade in self._trades], columns=columns)


def simulate_paper_predictions(
    decisions: Iterable[PaperSignal],
    prices: Iterable[float],
    *,
    initial_cash: float = 100_000.0,
    allocation: float = 1.0,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[PaperAccount, pd.DataFrame]:
    """Run deterministic virtual trading over precomputed signal/price pairs."""
    decisions = tuple(decisions)
    prices = tuple(float(price) for price in prices)
    if not decisions:
        raise ValueError("at least one paper signal is required")
    if len(decisions) != len(prices):
        raise ValueError("decisions and prices must have equal length")

    simulator = PaperTradingSimulator(
        initial_cash=initial_cash,
        allocation=allocation,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    for signal, price in zip(decisions, prices):
        simulator.step(signal, price)
    return simulator.account(prices[-1]), simulator.experience_frame()
