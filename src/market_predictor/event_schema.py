"""Schema for timestamped exogenous market events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class EventCategory(str, Enum):
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
    """An event known to the market from its publication timestamp onward.

    ``published_at`` is the time at which the information became available to
    market participants. Later confirmation or discovery timestamps must not
    replace it in a backtest.
    """

    event_id: str
    category: EventCategory
    published_at: datetime | str
    severity: float
    country: str | None = None
    entity: str | None = None
    sector: str | None = None
    duration_days: float = 0.0
    media_intensity: float = 0.0
    surprise: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.severity) <= 1.0:
            raise ValueError("severity must be between 0 and 1")
        if float(self.duration_days) < 0:
            raise ValueError("duration_days must be >= 0")
        if float(self.media_intensity) < 0:
            raise ValueError("media_intensity must be >= 0")
