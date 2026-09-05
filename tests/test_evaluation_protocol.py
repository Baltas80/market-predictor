from __future__ import annotations

import pytest

from market_predictor.evaluation_protocol import build_score, normalized_score


def test_normalized_score_is_bounded() -> None:
    assert normalized_score(5, lower=0, upper=10) == 0.5
    assert normalized_score(-1, lower=0, upper=10) == 0.0
    assert normalized_score(11, lower=0, upper=10) == 1.0


def test_score_groups_are_transparent() -> None:
    score = build_score({"accuracy": 0.6, "roc_auc": 0.8, "sharpe": 0.7, "total_return": 0.9, "stability": 0.5})
    assert score.predictive == 0.7
    assert score.financial == 0.8
    assert score.robustness == 0.5
    assert score.overall == pytest.approx(2.0 / 3.0)


def test_score_rejects_unnormalized_inputs() -> None:
    with pytest.raises(ValueError):
        build_score({"accuracy": 0.6, "roc_auc": 0.8, "sharpe": 1.2, "total_return": 0.9, "stability": 0.5})
