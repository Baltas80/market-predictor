"""Baseline model and time-series evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    roc_auc: float
    brier: float


def build_baseline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def fit_predict(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> tuple[Pipeline, pd.Series]:
    """Fit a fresh baseline and return P(up) for the test observations."""
    if y_train.nunique() < 2:
        raise ValueError("Training target must contain both classes")
    model = build_baseline()
    model.fit(x_train, y_train)
    probabilities = pd.Series(
        model.predict_proba(x_test)[:, 1], index=x_test.index, name="prob_up"
    )
    return model, probabilities


def evaluate(y_true: pd.Series, probabilities: pd.Series) -> Evaluation:
    """Evaluate classification and probability calibration metrics."""
    predictions = (probabilities >= 0.5).astype(int)
    roc_auc = float("nan") if y_true.nunique() < 2 else float(roc_auc_score(y_true, probabilities))
    return Evaluation(
        accuracy=float(accuracy_score(y_true, predictions)),
        roc_auc=roc_auc,
        brier=float(brier_score_loss(y_true, probabilities)),
    )
