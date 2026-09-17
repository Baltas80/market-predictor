"""Inference tools for paired, serially dependent OOS predictions.

These procedures are intended for validation samples that are already frozen by
an upstream chronological protocol. They do not select features, thresholds,
models, costs, or lockbox periods.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


Metric = Callable[[np.ndarray, np.ndarray], float]


def _validate_inputs(y_true: Sequence[int], prob_a: Sequence[float], prob_b: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(prob_a, dtype=float)
    b = np.asarray(prob_b, dtype=float)
    if y.ndim != 1 or a.ndim != 1 or b.ndim != 1:
        raise ValueError("y_true and prediction arrays must be one-dimensional")
    if not (len(y) == len(a) == len(b)):
        raise ValueError("paired arrays must have equal length")
    if len(y) < 2:
        raise ValueError("at least two paired observations are required")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only binary 0/1 labels")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("prediction probabilities must be finite")
    if ((a < 0) | (a > 1) | (b < 0) | (b > 1)).any():
        raise ValueError("prediction probabilities must lie in [0, 1]")
    return y, a, b


def accuracy_metric(y_true: np.ndarray, prob_up: np.ndarray) -> float:
    return float(np.mean((prob_up >= 0.5).astype(int) == y_true))


def brier_metric(y_true: np.ndarray, prob_up: np.ndarray) -> float:
    return float(np.mean((prob_up - y_true) ** 2))


def roc_auc_metric(y_true: np.ndarray, prob_up: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, prob_up))


def _resolve_metric(metric: str | Metric) -> Metric:
    if callable(metric):
        return metric
    metrics: dict[str, Metric] = {
        "accuracy": accuracy_metric,
        "brier": brier_metric,
        "roc_auc": roc_auc_metric,
    }
    try:
        return metrics[metric]
    except KeyError as exc:
        raise ValueError(f"Unsupported metric: {metric}") from exc


def _circular_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if block_length < 1 or block_length > n:
        raise ValueError("block_length must be between 1 and the sample size")
    indices: list[int] = []
    while len(indices) < n:
        start = int(rng.integers(0, n))
        take = min(block_length, n - len(indices))
        indices.extend((start + np.arange(take)) % n)
    return np.asarray(indices, dtype=int)


def paired_block_bootstrap_difference(
    y_true: Sequence[int],
    prob_a: Sequence[float],
    prob_b: Sequence[float],
    *,
    metric: str | Metric = "accuracy",
    block_length: int = 20,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int | str]:
    """Estimate B-A metric difference and a serial-dependence-aware CI.

    A circular moving-block bootstrap resamples the same timestamps for both
    methods, preserving their pairing and short-range temporal dependence.
    """
    y, a, b = _validate_inputs(y_true, prob_a, prob_b)
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be >= 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    fn = _resolve_metric(metric)
    observed_a = fn(y, a)
    observed_b = fn(y, b)
    observed = observed_b - observed_a
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    valid = 0
    for _ in range(n_bootstrap):
        idx = _circular_block_indices(len(y), block_length, rng)
        value_a = fn(y[idx], a[idx])
        value_b = fn(y[idx], b[idx])
        if np.isfinite(value_a) and np.isfinite(value_b):
            samples[valid] = value_b - value_a
            valid += 1
    if valid < max(100, n_bootstrap // 2):
        raise RuntimeError("too few valid bootstrap replicates")
    samples = samples[:valid]
    alpha = 1.0 - confidence
    low, high = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "metric": metric if isinstance(metric, str) else getattr(metric, "__name__", "callable"),
        "block_length": int(block_length),
        "bootstrap_replicates": int(valid),
        "confidence": float(confidence),
        "a_estimate": float(observed_a),
        "b_estimate": float(observed_b),
        "difference_b_minus_a": float(observed),
        "ci_low": float(low),
        "ci_high": float(high),
        "seed": int(seed),
    }


def paired_block_swap_test(
    y_true: Sequence[int],
    prob_a: Sequence[float],
    prob_b: Sequence[float],
    *,
    metric: str | Metric = "accuracy",
    block_length: int = 20,
    n_permutations: int = 5000,
    seed: int = 0,
) -> dict[str, float | int | str]:
    """Test paired A/B exchangeability by swapping labels at block level.

    Whole blocks are switched between A and B, preserving local serial
    structure while generating a null distribution under paired exchangeability.
    The returned p-value is two-sided.
    """
    y, a, b = _validate_inputs(y_true, prob_a, prob_b)
    if n_permutations < 100:
        raise ValueError("n_permutations must be >= 100")
    if block_length < 1 or block_length > len(y):
        raise ValueError("block_length must be between 1 and the sample size")
    fn = _resolve_metric(metric)
    observed = fn(y, b) - fn(y, a)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    n = len(y)
    blocks = [np.arange(start, min(start + block_length, n)) for start in range(0, n, block_length)]
    for i in range(n_permutations):
        swapped_a = a.copy()
        swapped_b = b.copy()
        for block in blocks:
            if bool(rng.integers(0, 2)):
                swapped_a[block], swapped_b[block] = b[block], a[block]
        value = fn(y, swapped_b) - fn(y, swapped_a)
        null[i] = 0.0 if not np.isfinite(value) else value
    p_value = (1.0 + float(np.sum(np.abs(null) >= abs(observed)))) / (n_permutations + 1.0)
    return {
        "metric": metric if isinstance(metric, str) else getattr(metric, "__name__", "callable"),
        "block_length": int(block_length),
        "permutations": int(n_permutations),
        "observed_difference_b_minus_a": float(observed),
        "p_value_two_sided": float(p_value),
        "seed": int(seed),
    }


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values in input order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p_values must be finite and lie in [0, 1]")
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted
