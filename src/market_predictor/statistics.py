"""Deterministic statistical robustness utilities for OOS predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class ClassificationSummary:
    observations: int
    accuracy: float
    roc_auc: float
    brier: float


def summarize_predictions(predictions: pd.DataFrame) -> ClassificationSummary:
    """Return core OOS metrics without fitting or modifying the predictions."""
    required = {"actual", "prob_up"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    y = pd.to_numeric(predictions["actual"], errors="coerce")
    p = pd.to_numeric(predictions["prob_up"], errors="coerce")
    valid = y.notna() & p.notna()
    if not valid.any():
        raise ValueError("No valid predictions")
    y = y.loc[valid].astype(int)
    p = p.loc[valid].clip(0.0, 1.0)
    pred = (p >= 0.5).astype(int)
    auc = float("nan") if y.nunique() < 2 else float(roc_auc_score(y, p))
    return ClassificationSummary(len(y), float(accuracy_score(y, pred)), auc, float(brier_score_loss(y, p)))


def bootstrap_metric_ci(
    y_true: pd.Series,
    probabilities: pd.Series,
    *,
    metric: str = "accuracy",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Compute a reproducible percentile bootstrap CI for an OOS metric.

    The bootstrap resamples prediction rows, not model training folds. It is a
    descriptive uncertainty estimate and should not be interpreted as a new
    backtest or as independence-adjusted financial inference.
    """
    if metric not in {"accuracy", "brier", "roc_auc"}:
        raise ValueError("metric must be accuracy, brier, or roc_auc")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be >= 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(probabilities, errors="coerce")
    valid = y.notna() & p.notna()
    y = y.loc[valid].astype(int).to_numpy()
    p = p.loc[valid].clip(0.0, 1.0).to_numpy()
    if len(y) < 2:
        raise ValueError("At least two valid predictions are required")

    def score(yy: np.ndarray, pp: np.ndarray) -> float:
        if metric == "accuracy":
            return float(accuracy_score(yy, pp >= 0.5))
        if metric == "brier":
            return float(brier_score_loss(yy, pp))
        if np.unique(yy).size < 2:
            return float("nan")
        return float(roc_auc_score(yy, pp))

    point = score(y, p)
    rng = np.random.default_rng(random_state)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y), size=len(y))
        value = score(y[indices], p[indices])
        if np.isfinite(value):
            samples.append(value)
    if not samples:
        raise ValueError("Bootstrap produced no valid metric samples")
    alpha = (1.0 - confidence) / 2.0
    return point, float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def period_stability(predictions: pd.DataFrame, periods: int = 4) -> pd.DataFrame:
    """Evaluate prediction metrics separately across chronological blocks."""
    if periods < 2:
        raise ValueError("periods must be >= 2")
    if not isinstance(predictions.index, pd.DatetimeIndex):
        raise TypeError("predictions must have a DatetimeIndex")
    ordered = predictions.sort_index()
    chunks = np.array_split(np.arange(len(ordered)), periods)
    rows: list[dict[str, float | int]] = []
    for number, indices in enumerate(chunks, start=1):
        if len(indices) == 0:
            continue
        summary = summarize_predictions(ordered.iloc[indices])
        rows.append({"period": number, "observations": summary.observations, "accuracy": summary.accuracy, "roc_auc": summary.roc_auc, "brier": summary.brier})
    return pd.DataFrame(rows)
