"""Paper-trading adapter for the frozen A/B/C prediction family."""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .experiments import ExperimentResult
from .paper_trading import PaperSignal, PaperTradingSimulator


def signals_from_predictions(
    predictions: pd.DataFrame,
    *,
    probability_threshold: float = 0.55,
) -> tuple[PaperSignal, ...]:
    """Convert OOS probabilities into auditable virtual-trading decisions.

    The threshold is an explicit protocol input. It is never inferred from OOS
    outcomes, so this adapter cannot tune itself against the evaluation set.
    """
    if not 0.5 < probability_threshold < 1:
        raise ValueError("probability_threshold must be in (0.5, 1)")
    required = {"prob_up"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    if not predictions.index.is_unique or not predictions.index.is_monotonic_increasing:
        raise ValueError("prediction index must be unique and chronological")

    signals: list[PaperSignal] = []
    for timestamp, row in predictions.iterrows():
        probability = float(row["prob_up"])
        if not 0 <= probability <= 1:
            raise ValueError("prob_up must be in [0,1]")
        position = 1 if probability >= probability_threshold else -1 if probability <= 1 - probability_threshold else 0
        signals.append(
            PaperSignal(
                signal_id=f"paper-{timestamp.isoformat()}",
                decision_time=timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp,
                probability_up=probability,
                confidence=abs(probability - 0.5) * 2,
                position=position,
                expected_risk=1.0 - abs(probability - 0.5) * 2,
            )
        )
    return tuple(signals)


def simulate_abc_paper_trading(
    results: list[ExperimentResult],
    prices: pd.Series,
    *,
    initial_cash: float = 100_000.0,
    allocation: float = 1.0,
    probability_threshold: float = 0.55,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> Mapping[str, tuple[float, pd.DataFrame]]:
    """Run identical virtual-account rules for A/B/C on their shared OOS index."""
    if not results:
        raise ValueError("at least one experiment result is required")
    reference_index = results[0].predictions.index
    if not reference_index.equals(prices.index):
        raise ValueError("prices must use the exact shared OOS prediction index")

    output: dict[str, tuple[float, pd.DataFrame]] = {}
    for result in results:
        if not result.predictions.index.equals(reference_index):
            raise ValueError("A/B/C predictions must use the exact shared OOS index")
        simulator = PaperTradingSimulator(
            initial_cash=initial_cash,
            allocation=allocation,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )
        signals = signals_from_predictions(result.predictions, probability_threshold=probability_threshold)
        for signal, price in zip(signals, prices.to_numpy()):
            simulator.step(signal, float(price))
        equity = simulator.account(float(prices.iloc[-1])).equity
        output[result.name] = (equity, simulator.experience_frame())
    return output
