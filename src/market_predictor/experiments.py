"""Comparable out-of-sample experiments for feature ablations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import Fold, walk_forward_classification
from .metrics import summarize_predictions

TECHNICAL = [
    "return_1d", "return_5d", "volatility_20d", "price_to_sma20",
    "volume_change", "range_pct",
]


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    features: tuple[str, ...]
    evaluations: list
    predictions: pd.DataFrame


def run_feature_ablation(
    data: pd.DataFrame,
    folds: list[Fold],
    *,
    macro_features: list[str] | None = None,
    geopolitical_features: list[str] | None = None,
    target: str = "target",
) -> list[ExperimentResult]:
    """Run nested feature sets on identical chronological folds.

    Keeping folds identical is essential: differences between models should
    come from information sets, not from different train/test periods.
    """
    macro_features = macro_features or []
    geopolitical_features = geopolitical_features or []
    specs = [
        ("technical", TECHNICAL),
        ("technical_macro", TECHNICAL + macro_features),
        ("technical_macro_geopolitical", TECHNICAL + macro_features + geopolitical_features),
    ]
    results: list[ExperimentResult] = []
    for name, features in specs:
        missing = [column for column in features + [target] if column not in data.columns]
        if missing:
            raise ValueError(f"Missing columns for {name}: {missing}")
        evaluations, predictions = walk_forward_classification(data, features, target, folds)
        results.append(ExperimentResult(name, tuple(features), evaluations, predictions))
    return results


def summarize_experiments(results: list[ExperimentResult]) -> pd.DataFrame:
    """Aggregate metrics over all OOS predictions, not over fold means."""
    rows = []
    for result in results:
        summary = summarize_predictions(result.predictions)
        rows.append({
            "experiment": result.name,
            "features": len(result.features),
            **summary,
            "folds": len(result.evaluations),
            "predictions": len(result.predictions),
        })
    return pd.DataFrame(rows)
