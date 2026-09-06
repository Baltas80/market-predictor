"""Load historical market events from auditable tabular sources."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .event_schema import EventCategory, MarketEvent

REQUIRED_COLUMNS = {"event_id", "category", "published_at", "severity"}
OPTIONAL_COLUMNS = {
    "event_time",
    "available_at",
    "country",
    "entity",
    "sector",
    "duration_days",
    "media_intensity",
    "surprise",
}


def load_events_csv(path: str | Path) -> list[MarketEvent]:
    """Load events from CSV while preserving occurrence/publication/availability times.

    Required columns are ``event_id``, ``category``, ``published_at`` and
    ``severity``. If ``event_time`` and/or ``available_at`` are present they
    are passed through unchanged so downstream feature generation can enforce
    the information cutoff. Duplicate event IDs are rejected.
    """
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")
    if frame["event_id"].duplicated().any():
        duplicates = frame.loc[frame["event_id"].duplicated(), "event_id"].tolist()
        raise ValueError(f"Duplicate event_id values: {duplicates}")

    events: list[MarketEvent] = []
    for row_number, row in frame.iterrows():
        try:
            category = EventCategory(str(row["category"]))
            published_at = pd.Timestamp(row["published_at"])
            if pd.isna(published_at):
                raise ValueError("published_at is missing")
            event = MarketEvent(
                event_id=str(row["event_id"]),
                category=category,
                published_at=published_at,
                severity=float(row["severity"]),
                event_time=_optional_timestamp(row.get("event_time")),
                available_at=_optional_timestamp(row.get("available_at")),
                country=_optional_text(row.get("country")),
                entity=_optional_text(row.get("entity")),
                sector=_optional_text(row.get("sector")),
                duration_days=_optional_float(row.get("duration_days"), 0.0),
                media_intensity=_optional_float(row.get("media_intensity"), 0.0),
                surprise=_optional_float(row.get("surprise"), 0.0),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid event at CSV row {row_number + 2}: {exc}") from exc
        events.append(event)

    return sorted(events, key=lambda event: pd.Timestamp(event.published_at))


def _optional_timestamp(value: object) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    return timestamp


def _optional_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _optional_float(value: object, default: float) -> float:
    if value is None or pd.isna(value):
        return default
    return float(value)
