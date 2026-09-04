"""Chronological baseline pipeline."""

from __future__ import annotations

import pandas as pd

from .features import add_market_features, make_target
from .model import Evaluation, evaluate, fit_predict

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "volatility_20d",
    "price_to_sma20",
    "volume_change",
    "range_pct",
]


def run_baseline(df: pd.DataFrame, horizon: int = 5, train_fraction: float = 0.8) -> tuple[Evaluation, pd.Series]:
    """Train on the earliest observations and test on the latest observations.

    No random split is used, preventing future observations from entering training.
    """
    if not 0.5 <= train_fraction < 1:
        raise ValueError("train_fraction must be >= 0.5 and < 1")

    data = add_market_features(df)
    data["target"] = make_target(data, horizon=horizon)
    data = data.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()

    split = int(len(data) * train_fraction)
    if split <= 0 or split >= len(data):
        raise ValueError("Not enough observations for train/test split")

    train = data.iloc[:split]
    test = data.iloc[split:]
    model, probabilities = fit_predict(
        train[FEATURE_COLUMNS], train["target"].astype(int), test[FEATURE_COLUMNS]
    )
    del model
    metrics = evaluate(test["target"].astype(int), probabilities)
    return metrics, probabilities
