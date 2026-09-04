"""Classification and probability diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    """Compute OOS diagnostics from actual labels and predicted probabilities."""
    required = {"actual", "prob_up"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    frame = predictions.dropna(subset=["actual", "prob_up"]).copy()
    if frame.empty:
        return {"accuracy": float("nan"), "brier": float("nan"), "roc_auc": float("nan"), "base_rate": float("nan")}
    actual = frame["actual"].astype(int).to_numpy()
    prob = frame["prob_up"].clip(0.0, 1.0).to_numpy()
    predicted = (prob >= 0.5).astype(int)
    accuracy = float(np.mean(predicted == actual))
    brier = float(np.mean((prob - actual) ** 2))
    if len(np.unique(actual)) == 2:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(actual, prob))
    else:
        auc = float("nan")
    return {"accuracy": accuracy, "brier": brier, "roc_auc": auc, "base_rate": float(np.mean(actual))}
