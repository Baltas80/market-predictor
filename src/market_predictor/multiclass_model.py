"""Multiclass directional model and evaluation for research experiments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .directional_target import DIRECTION_CLASSES


@dataclass(frozen=True)
class MulticlassEvaluation:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    down_precision: float
    down_recall: float
    down_f1: float
    flat_precision: float
    flat_recall: float
    flat_f1: float
    up_precision: float
    up_recall: float
    up_f1: float


def build_multiclass_baseline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def fit_predict_multiclass(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Fit a fresh multiclass model and return P(down/flat/up) and labels."""
    y_train = y_train.astype(int)
    if set(y_train.unique()) != set(DIRECTION_CLASSES):
        raise ValueError(
            "multiclass training target must contain down, flat and up classes"
        )

    model = build_multiclass_baseline()
    model.fit(x_train, y_train)
    probabilities = pd.DataFrame(
        model.predict_proba(x_test),
        index=x_test.index,
        columns=[int(value) for value in model.classes_],
    )
    if list(model.classes_) != list(DIRECTION_CLASSES):
        raise ValueError("model classes do not match the canonical direction order")
    probabilities.columns = ["prob_down", "prob_flat", "prob_up"]
    predictions = pd.Series(
        model.predict(x_test),
        index=x_test.index,
        name="predicted_direction",
        dtype=int,
    )
    return model, probabilities, predictions


def evaluate_multiclass(
    y_true: pd.Series,
    probabilities: pd.DataFrame,
    predictions: pd.Series,
) -> MulticlassEvaluation:
    """Evaluate all three directions separately."""
    y_true = y_true.astype(int)
    predictions = predictions.astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=list(DIRECTION_CLASSES),
        zero_division=0,
    )
    return MulticlassEvaluation(
        accuracy=float(accuracy_score(y_true, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, predictions)),
        macro_f1=float(f1_score(y_true, predictions, labels=list(DIRECTION_CLASSES), average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, predictions, labels=list(DIRECTION_CLASSES), average="weighted", zero_division=0)),
        down_precision=float(precision[0]),
        down_recall=float(recall[0]),
        down_f1=float(f1[0]),
        flat_precision=float(precision[1]),
        flat_recall=float(recall[1]),
        flat_f1=float(f1[1]),
        up_precision=float(precision[2]),
        up_recall=float(recall[2]),
        up_f1=float(f1[2]),
    )


def summarize_confusion(y_true: pd.Series, predictions: pd.Series) -> np.ndarray:
    """Return a fixed {-1,0,+1} confusion matrix."""
    return confusion_matrix(
        y_true.astype(int),
        predictions.astype(int),
        labels=list(DIRECTION_CLASSES),
    )
