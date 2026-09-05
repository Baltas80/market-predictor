"""Strict schema for AI-derived event features used by experiment D."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class AIEventSignal:
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
        for name, value in (("severity", self.severity), ("surprise", self.surprise), ("market_relevance", self.market_relevance), ("confidence", self.confidence)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.available_at.tzinfo is None:
            raise ValueError("available_at must be timezone-aware")

    def as_features(self) -> dict[str, float | str | datetime]:
        self.validate()
        return {"event_id": self.event_id, "source_id": self.source_id, "category": self.category, "ai_severity": float(self.severity), "ai_surprise": float(self.surprise), "ai_market_relevance": float(self.market_relevance), "ai_confidence": float(self.confidence), "available_at": self.available_at}
