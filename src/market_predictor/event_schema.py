"""Structured taxonomy for exogenous market-impacting events."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EventCategory(str, Enum):
    """High-level event families used by the geopolitical/event model."""

    WAR_CONFLICT = "war_conflict"
    POLITICAL_CRISIS = "political_crisis"
    CORRUPTION = "corruption"
    CORPORATE_SCANDAL = "corporate_scandal"
    POLITICAL_SCANDAL = "political_scandal"
    FINANCIAL_FRAUD = "financial_fraud"
    SANCTIONS = "sanctions"
    REGULATION = "regulation"
    TERRORISM = "terrorism"
    SOCIAL_UNREST = "social_unrest"
    ECONOMIC_CRISIS = "economic_crisis"
    NATURAL_DISASTER = "natural_disaster"
    PUBLIC_HEALTH = "public_health"


@dataclass(frozen=True)
class MarketEvent:
    """An event observation with publication-time semantics.

    ``published_at`` is the timestamp at which the information became
    available to the market. Backtests must use this timestamp rather than a
    later confirmation, discovery, or retrospective database timestamp.
    """

    event_id: str
    category: EventCategory
    published_at: object
    severity: float
    country: str | None = None
    entity: str | None = None
    sector: str | None = None
    duration_days: float = 0.0
    media_intensity: float = 0.0
    surprise: float = 0.0

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id cannot be empty")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be between 0 and 1")
        if self.duration_days < 0:
            raise ValueError("duration_days cannot be negative")
        if self.media_intensity < 0:
            raise ValueError("media_intensity cannot be negative")
