"""Explicit research scorecard for model and financial evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class EvaluationScore:
    """Separate predictive, financial and robustness evidence."""

    predictive: float
    financial: float
    robustness: float

    @property
    def overall(self) -> float:
        """Equal-weight summary; components must already be normalized."""
        return (self.predictive + self.financial + self.robustness) / 3.0


def normalized_score(value: float, *, lower: float, upper: float) -> float:
    """Map a metric into [0, 1] without changing its interpretation."""
    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    return max(0.0, min(1.0, (float(value) - lower) / (upper - lower)))


def build_score(
    metrics: Mapping[str, float],
    *,
    predictive_keys: tuple[str, ...] = ("accuracy", "roc_auc"),
    financial_keys: tuple[str, ...] = ("sharpe", "total_return"),
    robustness_keys: tuple[str, ...] = ("stability",),
) -> EvaluationScore:
    """Build a transparent score from pre-normalized [0,1] components."""
    groups = (predictive_keys, financial_keys, robustness_keys)
    values: list[float] = []
    for keys in groups:
        selected = [float(metrics[key]) for key in keys if key in metrics]
        if not selected:
            raise ValueError("Each score group needs at least one metric")
        if any(not 0.0 <= value <= 1.0 for value in selected):
            raise ValueError("score inputs must be normalized to [0, 1]")
        values.append(sum(selected) / len(selected))
    return EvaluationScore(*values)
