"""Auditable AI overlay contract; AI cannot supply targets or future outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

_FORBIDDEN = {"target", "y_true", "outcome", "label", "prediction", "predicted_return", "future_return"}


@dataclass(frozen=True)
class AIEventSignalV3:
    event_id: str
    source_id: str
    category: str
    severity: float
    surprise: float
    market_relevance: float
    confidence: float
    available_at: datetime

    def validate(self) -> None:
        if not self.event_id or not self.source_id or not self.category:
            raise ValueError("event_id, source_id and category are required")
        for name in ("severity", "surprise", "market_relevance", "confidence"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.available_at.tzinfo is None:
            raise ValueError("available_at must be timezone-aware")

    def as_features(self) -> dict[str, object]:
        self.validate()
        return {
            "ai_severity": self.severity,
            "ai_surprise": self.surprise,
            "ai_market_relevance": self.market_relevance,
            "ai_confidence": self.confidence,
            "available_at": self.available_at,
        }


def assert_ai_available(signal: AIEventSignalV3, decision_time: datetime) -> None:
    """Reject any AI information that became available after the decision."""
    signal.validate()
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    if signal.available_at > decision_time:
        raise ValueError("AI signal is not available at decision time")


def validate_ai_feature_names(feature_names: list[str] | tuple[str, ...]) -> None:
    """Reject names that could smuggle target/outcome information into AI features."""
    lowered = {name.lower() for name in feature_names}
    forbidden = sorted(lowered.intersection(_FORBIDDEN))
    if forbidden:
        raise ValueError(f"forbidden AI feature names: {forbidden}")
