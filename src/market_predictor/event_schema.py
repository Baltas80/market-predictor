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
    """Event with explicit occurrence, publication and information-availability times.

    ``available_at`` is the only timestamp permitted to gate model information.
    ``published_at`` may be unknown when the source exposes an independent
    availability timestamp, as GDELT does through DATEADDED.
    """

    event_id: str
    category: EventCategory
    published_at: datetime | str | None
    severity: float
    event_time: datetime | str | None = None
    available_at: datetime | str | None = None
    country: str | None = None
    entity: str | None = None
    sector: str | None = None
    duration_days: float = 0.0
    media_intensity: float = 0.0
    surprise: float = 0.0

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id is required")
        if not 0.0 <= float(self.severity) <= 1.0:
            raise ValueError("severity must be between 0 and 1")
        if float(self.duration_days) < 0:
            raise ValueError("duration_days must be >= 0")
        if float(self.media_intensity) < 0:
            raise ValueError("media_intensity must be >= 0")

        published = (
            datetime.fromisoformat(str(self.published_at).replace("Z", "+00:00"))
            if isinstance(self.published_at, str)
            else self.published_at
        )
        available = (
            datetime.fromisoformat(str(self.available_at).replace("Z", "+00:00"))
            if isinstance(self.available_at, str)
            else self.available_at
        )
        if published is None and available is None:
            raise ValueError("published_at or available_at is required")
        if available is None:
            available = published
            object.__setattr__(self, "available_at", published)
        if available is not None and available.tzinfo is None:
            raise ValueError("available_at must be timezone-aware")
        if published is not None and published.tzinfo is None:
            raise ValueError("published_at must be timezone-aware")
        if available is not None and published is not None and available < published:
            raise ValueError("available_at cannot precede published_at")
        if self.event_time is not None:
            event_time = (
                datetime.fromisoformat(str(self.event_time).replace("Z", "+00:00"))
                if isinstance(self.event_time, str)
                else self.event_time
            )
            if event_time.tzinfo is None:
                raise ValueError("event_time must be timezone-aware")
            if available is not None and event_time > available:
                raise ValueError("event_time cannot be after available_at")
            if published is not None and event_time > published:
                raise ValueError("event_time cannot be after published_at")
