"""Common-fold feature-group ablation for incremental-value analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import pandas as pd


@dataclass(frozen=True)
class AblationSpec:
    name: str
    feature_columns: tuple[str, ...]


def build_ablation_specs(
    *,
    technical: tuple[str, ...],
    macro: tuple[str, ...],
    events: tuple[str, ...],
) -> tuple[AblationSpec, ...]:
    """Build A/B/C feature sets without changing the observation protocol."""
    if not technical:
        raise ValueError("technical features cannot be empty")
    return (
        AblationSpec("A", technical),
        AblationSpec("B", technical + macro),
        AblationSpec("C", technical + macro + events),
    )


def run_common_fold_ablation(
    data: pd.DataFrame,
    specs: tuple[AblationSpec, ...],
    evaluator: Callable[[pd.DataFrame, tuple[str, ...]], Mapping[str, float]],
) -> pd.DataFrame:
    """Evaluate feature groups through the same supplied chronological evaluator."""
    if not specs:
        raise ValueError("at least one ablation specification is required")
    rows: list[dict[str, float | str]] = []
    for spec in specs:
        missing = [column for column in spec.feature_columns if column not in data.columns]
        if missing:
            raise ValueError(f"missing features for {spec.name}: {missing}")
        metrics = dict(evaluator(data, spec.feature_columns))
        rows.append({"experiment": spec.name, **metrics})
    return pd.DataFrame(rows)
