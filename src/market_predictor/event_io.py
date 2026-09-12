"""Load historical market events from auditable tabular sources."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .event_schema import EventCategory, MarketEvent

REQUIRED_COLUMNS = {"event_id", "category", "severity"}
OPTIONAL_COLUMNS = {
    "event_time",
    "published_at",
    "available_at",
    "country",
    "entity",
    "sector",
    "duration_days",
    "media_intensity",
    "surprise",
}


def load_events_csv(path: str | Path) -> list[MarketEvent]:
    """Load events while keeping publication and availability semantically distinct.

    Every event must provide either ``available_at`` or ``published_at``. A
    source may legitimately omit publication time when it exposes an explicit
    information-availability timestamp (for example, GDELT DATEADDED).
    """
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")
    if not ({"published_at", "available_at"} & set(frame.columns)):
        raise ValueError("Event CSV requires published_at or available_at")
    if frame["event_id"].duplicated().any():
        duplicates = frame.loc[frame["event_id"].duplicated(), "event_id"].tolist()
        raise ValueError(f"Duplicate event_id values: {duplicates}")

    events: list[MarketEvent] = []
    for row_number, row in frame.iterrows():
        try:
            event_id = _required_text(row.get("event_id"), "event_id")
            category = EventCategory(str(row["category"]))
            published_at = _optional_timestamp(row.get("published_at"))
            available_at = _optional_timestamp(row.get("available_at"))
            if published_at is None and available_at is None:
                raise ValueError("published_at and available_at are both missing")
            event = MarketEvent(
                event_id=event_id,
                category=category,
                published_at=published_at,
                severity=float(row["severity"]),
                event_time=_optional_timestamp(row.get("event_time")),
                available_at=available_at,
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

    return sorted(events, key=lambda event: pd.Timestamp(event.available_at))


def _required_text(value: object, name: str) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError(f"{name} is missing")
    return str(value)


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
