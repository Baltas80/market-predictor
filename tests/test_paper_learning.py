from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from market_predictor.paper_learning import candidate_probability, train_candidate


UTC = timezone.utc


def experiences(rows: int = 40) -> pd.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    values = []
    for i in range(rows):
        probability = 0.8 if i % 2 == 0 else 0.2
        position = 1 if i % 2 == 0 else -1
        realized = 0.02 if i % 2 == 0 else -0.01
        values.append(
            {
                "decision_time": start + timedelta(days=i),
                "outcome_time": start + timedelta(days=i + 1),
                "probability_up": probability,
                "confidence": 0.6,
                "position": position,
                "notional": 100_000.0,
                "realized_return": realized,
            }
        )
    return pd.DataFrame(values)


def test_candidate_uses_newest_experiences_as_validation():
    frame = experiences(40)
    candidate = train_candidate(frame, min_train_rows=30, validation_fraction=0.25)
    assert candidate.report.train_rows == 30
    assert candidate.report.validation_rows == 10
    assert candidate.report.candidate_id.startswith("paper-meta-v1-30-10-")


def test_candidate_can_score_decision_time_features():
    candidate = train_candidate(experiences(40), min_train_rows=30)
    scored = candidate_probability(candidate, experiences(3))
    assert len(scored) == 3
    assert scored.between(0, 1).all()


def test_learning_rejects_future_outcome_leakage():
    frame = experiences(40)
    frame.loc[10, "outcome_time"] = frame.loc[10, "decision_time"]
    with pytest.raises(ValueError, match="outcome_time must follow decision_time"):
        train_candidate(frame, min_train_rows=30)


def test_learning_requires_enough_rows():
    with pytest.raises(ValueError, match="not enough resolved experiences"):
        train_candidate(experiences(10), min_train_rows=30)


def test_learning_requires_both_training_outcome_classes():
    frame = experiences(40)
    frame["realized_return"] = 0.01
    with pytest.raises(ValueError, match="both outcome classes"):
        train_candidate(frame, min_train_rows=30)


def test_learning_rejects_duplicate_decision_times():
    frame = experiences(40)
    frame.loc[1, "decision_time"] = frame.loc[0, "decision_time"]
    with pytest.raises(ValueError, match="decision_time must be unique"):
        train_candidate(frame, min_train_rows=30)
