from __future__ import annotations

import pandas as pd
import pytest

from market_predictor.statistics import bootstrap_metric_ci, period_stability, summarize_predictions


def sample_predictions(n: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    actual = pd.Series([0, 1] * (n // 2), index=idx)
    prob = actual.astype(float) * 0.8 + (1 - actual.astype(float)) * 0.2
    return pd.DataFrame({"actual": actual, "prob_up": prob}, index=idx)


def test_summary_metrics() -> None:
    summary = summarize_predictions(sample_predictions())
    assert summary.observations == 40
    assert summary.accuracy == 1.0
    assert summary.brier < 0.1
    assert summary.roc_auc == 1.0


def test_bootstrap_is_reproducible() -> None:
    frame = sample_predictions()
    first = bootstrap_metric_ci(frame.actual, frame.prob_up, n_bootstrap=200, random_state=42)
    second = bootstrap_metric_ci(frame.actual, frame.prob_up, n_bootstrap=200, random_state=42)
    assert first == second
    assert first[1] <= first[0] <= first[2]


def test_period_stability_preserves_chronological_blocks() -> None:
    result = period_stability(sample_predictions(), periods=4)
    assert len(result) == 4
    assert result["observations"].sum() == 40
    assert result["accuracy"].eq(1.0).all()


def test_invalid_metric_is_rejected() -> None:
    frame = sample_predictions()
    with pytest.raises(ValueError):
        bootstrap_metric_ci(frame.actual, frame.prob_up, metric="sharpe", n_bootstrap=200)
