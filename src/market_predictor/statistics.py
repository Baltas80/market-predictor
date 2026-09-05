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
    required = {"actual", "prob_up"}
    missing = required - set(predictions.columns)
    if missing: raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    y = pd.to_numeric(predictions["actual"], errors="coerce")
    p = pd.to_numeric(predictions["prob_up"], errors="coerce")
    valid = y.notna() & p.notna()
    if not valid.any(): raise ValueError("No valid predictions")
    y, p = y.loc[valid].astype(int), p.loc[valid].clip(0.0, 1.0)
    auc = float("nan") if y.nunique() < 2 else float(roc_auc_score(y, p))
    return ClassificationSummary(len(y), float(accuracy_score(y, p >= 0.5)), auc, float(brier_score_loss(y, p)))

def bootstrap_metric_ci(y_true, probabilities, *, metric="accuracy", n_bootstrap=2000, confidence=0.95, random_state=42):
    if metric not in {"accuracy", "brier", "roc_auc"}: raise ValueError("metric must be accuracy, brier, or roc_auc")
    if n_bootstrap < 100 or not 0 < confidence < 1: raise ValueError("invalid bootstrap configuration")
    y = pd.to_numeric(y_true, errors="coerce"); p = pd.to_numeric(probabilities, errors="coerce")
    valid = y.notna() & p.notna(); y = y.loc[valid].astype(int).to_numpy(); p = p.loc[valid].clip(0,1).to_numpy()
    if len(y) < 2: raise ValueError("At least two valid predictions are required")
    def score(yy, pp):
        if metric == "accuracy": return float(accuracy_score(yy, pp >= .5))
        if metric == "brier": return float(brier_score_loss(yy, pp))
        return float("nan") if np.unique(yy).size < 2 else float(roc_auc_score(yy, pp))
    point = score(y,p); rng=np.random.default_rng(random_state); samples=[]
    for _ in range(n_bootstrap):
        idx=rng.integers(0,len(y),size=len(y)); value=score(y[idx],p[idx])
        if np.isfinite(value): samples.append(value)
    if not samples: raise ValueError("Bootstrap produced no valid metric samples")
    alpha=(1-confidence)/2
    return point,float(np.quantile(samples,alpha)),float(np.quantile(samples,1-alpha))

def period_stability(predictions: pd.DataFrame, periods=4) -> pd.DataFrame:
    if periods < 2: raise ValueError("periods must be >= 2")
    if not isinstance(predictions.index,pd.DatetimeIndex): raise TypeError("predictions must have a DatetimeIndex")
    ordered=predictions.sort_index(); chunks=np.array_split(np.arange(len(ordered)),periods); rows=[]
    for number,indices in enumerate(chunks,1):
        if len(indices)==0: continue
        s=summarize_predictions(ordered.iloc[indices]); rows.append({"period":number,"observations":s.observations,"accuracy":s.accuracy,"roc_auc":s.roc_auc,"brier":s.brier})
    return pd.DataFrame(rows)
