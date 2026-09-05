"""Chronological model comparison and Champion/Challenger selection.

This module evaluates fresh candidate models on the same out-of-sample data.
It deliberately does not promote or mutate a production model.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelScore:
    name: str
    accuracy: float
    roc_auc: float
    brier: float


@dataclass(frozen=True)
class ChallengerDecision:
    champion: str
    challenger: str
    challenger_is_better: bool
    reason: str


def build_model(name: str):
    """Build a deterministic candidate model by name."""
    if name == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ])
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            random_state=42, n_jobs=1,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42,
        )
    raise ValueError(f"unknown model: {name}")


def compare_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    model_names: tuple[str, ...] = ("logistic", "random_forest", "hist_gradient_boosting"),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit all models on identical training data and score identical OOS data."""
    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("feature/target lengths must match")
    if not x_train.index.equals(y_train.index) or not x_test.index.equals(y_test.index):
        raise ValueError("feature/target indexes must match")
    if not x_test.index.is_monotonic_increasing or not x_test.index.is_unique:
        raise ValueError("OOS index must be unique and chronological")
    if y_train.nunique() < 2:
        raise ValueError("training target must contain both classes")
    if not model_names:
        raise ValueError("at least one model is required")

    scores: list[ModelScore] = []
    fitted: dict[str, object] = {}
    for name in model_names:
        model = build_model(name)
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        auc = float("nan") if y_test.nunique() < 2 else float(roc_auc_score(y_test, probabilities))
        scores.append(ModelScore(
            name=name,
            accuracy=float(accuracy_score(y_test, predictions)),
            roc_auc=auc,
            brier=float(brier_score_loss(y_test, probabilities)),
        ))
        fitted[name] = model

    table = pd.DataFrame([s.__dict__ for s in scores]).sort_values(
        ["brier", "accuracy", "name"], ascending=[True, False, True], kind="stable"
    ).reset_index(drop=True)
    return table, fitted


def decide_challenger(
    champion: ModelScore,
    challenger: ModelScore,
    *,
    min_accuracy_gain: float = 0.0,
    max_brier_increase: float = 0.0,
) -> ChallengerDecision:
    """Make a conservative recommendation; never performs promotion itself."""
    if min_accuracy_gain < 0 or max_brier_increase < 0:
        raise ValueError("decision tolerances must be non-negative")
    accuracy_gain = challenger.accuracy - champion.accuracy
    brier_change = challenger.brier - champion.brier
    better = accuracy_gain >= min_accuracy_gain and brier_change <= max_brier_increase
    reason = (
        f"accuracy_gain={accuracy_gain:.6f}, brier_change={brier_change:.6f}; "
        f"promotion={'eligible' if better else 'rejected'} pending full backtest and gate"
    )
    return ChallengerDecision(
        champion=champion.name,
        challenger=challenger.name,
        challenger_is_better=better,
        reason=reason,
    )
