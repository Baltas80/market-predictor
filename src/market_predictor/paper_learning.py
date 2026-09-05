"""Safe learning loop for resolved paper-trading experiences.

The learner is a candidate generator only. It never edits a frozen model or the
final OOS lockbox. Training and validation are chronological and use disjoint
resolved experiences.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPERIENCE_FEATURES = (
    "probability_up",
    "confidence",
    "position",
    "entry_price",
    "notional",
)


@dataclass(frozen=True)
class LearningReport:
    """Audit record for a candidate-learning run."""

    train_rows: int
    validation_rows: int
    validation_accuracy: float
    validation_brier: float
    candidate_id: str


@dataclass(frozen=True)
class CandidateModel:
    """A newly trained paper candidate plus its chronological validation report."""

    model: Pipeline
    feature_names: tuple[str, ...]
    report: LearningReport


def _validate_experience_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(EXPERIENCE_FEATURES) | {"decision_time", "outcome_time", "realized_return"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing experience columns: {missing}")
    clean = frame.copy()
    clean["decision_time"] = pd.to_datetime(clean["decision_time"], utc=True)
    clean["outcome_time"] = pd.to_datetime(clean["outcome_time"], utc=True)
    if clean["decision_time"].isna().any() or clean["outcome_time"].isna().any():
        raise ValueError("experience timestamps cannot be null")
    if (clean["outcome_time"] <= clean["decision_time"]).any():
        raise ValueError("outcome_time must follow decision_time")
    if not clean["decision_time"].is_monotonic_increasing:
        clean = clean.sort_values("decision_time", kind="stable")
    if clean["decision_time"].duplicated().any():
        raise ValueError("decision_time must be unique for learning")
    if clean[list(EXPERIENCE_FEATURES) + ["realized_return"]].isna().any().any():
        raise ValueError("learning features and realized_return cannot be null")
    return clean.reset_index(drop=True)


def train_candidate(
    experiences: pd.DataFrame,
    *,
    min_train_rows: int = 30,
    validation_fraction: float = 0.25,
) -> CandidateModel:
    """Train a candidate meta-model from resolved paper outcomes.

    The target is whether the resolved virtual trade made a positive return.
    Only information recorded at decision time is used as an input feature.
    The newest chronological observations are held out for validation.
    """
    if min_train_rows < 4:
        raise ValueError("min_train_rows must be >= 4")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")

    frame = _validate_experience_frame(experiences)
    validation_rows = max(1, int(round(len(frame) * validation_fraction)))
    train_rows = len(frame) - validation_rows
    if train_rows < min_train_rows:
        raise ValueError("not enough resolved experiences for candidate training")

    train = frame.iloc[:train_rows]
    validation = frame.iloc[train_rows:]
    y_train = (train["realized_return"] > 0).astype(int)
    if y_train.nunique() < 2:
        raise ValueError("training experiences must contain both outcome classes")

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    model.fit(train[list(EXPERIENCE_FEATURES)], y_train)

    probabilities = model.predict_proba(validation[list(EXPERIENCE_FEATURES)])[:, 1]
    actual = (validation["realized_return"] > 0).astype(int).to_numpy()
    accuracy = float(((probabilities >= 0.5).astype(int) == actual).mean())
    brier = float(((probabilities - actual) ** 2).mean())

    candidate_id = (
        f"paper-meta-v1-{train_rows}-{validation_rows}-"
        f"{frame['decision_time'].iloc[train_rows - 1].isoformat()}"
    )
    report = LearningReport(
        train_rows=train_rows,
        validation_rows=validation_rows,
        validation_accuracy=accuracy,
        validation_brier=brier,
        candidate_id=candidate_id,
    )
    return CandidateModel(model=model, feature_names=EXPERIENCE_FEATURES, report=report)


def candidate_probability(candidate: CandidateModel, experiences: pd.DataFrame) -> pd.Series:
    """Score new, already-resolved or decision-time feature rows with a candidate."""
    missing = set(candidate.feature_names) - set(experiences.columns)
    if missing:
        raise ValueError(f"missing candidate features: {sorted(missing)}")
    values = candidate.model.predict_proba(experiences[list(candidate.feature_names)])[:, 1]
    return pd.Series(values, index=experiences.index, name="candidate_prob_positive_return")
