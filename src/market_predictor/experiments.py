"""Comparable out-of-sample experiments for feature ablations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import walk_forward_classification
from .model import Evaluation

TECHNICAL = [
    "return_1d", "return_5d", "volatility_20d", "price_to_sma20",
    "volume_change", "range_pct",
]


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    features: tuple[str, ...]
    evaluations: list[Evaluation]
    predictions: pd.DataFrame


def run_feature_ablation(
    data: pd.DataFrame,
    folds,
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
    """Return fold-weighted aggregate metrics for model comparison."""
    rows = []
    for result in results:
        if not result.evaluations:
            continue
        weights = [len(result.predictions) / len(result.evaluations)] * len(result.evaluations)
        del weights
        rows.append({
            "experiment": result.name,
            "features": len(result.features),
            "accuracy": sum(e.accuracy for e in result.evaluations) / len(result.evaluations),
            "roc_auc": _nanmean([e.roc_auc for e in result.evaluations]),
            "brier": _nanmean([e.brier for e in result.evaluations]),
            "predictions": len(result.predictions),
        })
    return pd.DataFrame(rows)


def _nanmean(values: list[float]) -> float:
    valid = [value for value in values if pd.notna(value)]
    return float(sum(valid) / len(valid)) if valid else float("nan")
