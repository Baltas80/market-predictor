"""Deterministic assembly and audit helpers for historical research data."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable
import hashlib
import json
from pathlib import Path

import pandas as pd

from .research_schema import validate_market_frame, validate_macro_frame, validate_event_frame, deduplicate_events, admitted_events
from .research_gate import assert_market_session_audit


@dataclass(frozen=True)
class SourceManifest:
    source_id: str
    source_type: str
    coverage_start: str
    coverage_end: str
    rows: int
    sha256: str
    retrieval_version: str
    source_uri: str = ""
    schema_version: str = "2026-09-05"
    availability_policy: str = ""

    def validate(self) -> None:
        if not self.source_id or not self.source_type or not self.sha256:
            raise ValueError("source manifest identity is required")
        if self.rows < 0:
            raise ValueError("source row count cannot be negative")
        if pd.Timestamp(self.coverage_end) < pd.Timestamp(self.coverage_start):
            raise ValueError("source coverage end cannot precede start")
        if not self.schema_version:
            raise ValueError("source manifest schema_version is required")


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash a canonical tabular representation independent of row index objects."""
    normalized = frame.copy()
    normalized = normalized.sort_index() if isinstance(normalized.index, pd.DatetimeIndex) else normalized.sort_values(list(normalized.columns)).reset_index(drop=True)
    payload = normalized.to_csv(index=True, date_format="%Y-%m-%dT%H:%M:%S%z", lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_source_manifest(frame: pd.DataFrame, *, source_id: str, source_type: str, retrieval_version: str = "1", source_uri: str = "", schema_version: str = "2026-09-05", availability_policy: str = "") -> SourceManifest:
    if frame.empty:
        raise ValueError("cannot build a coverage manifest from an empty frame")
    if isinstance(frame.index, pd.DatetimeIndex):
        start, end = frame.index.min(), frame.index.max()
    else:
        candidates = [c for c in ("available_at", "published_at", "observation_date", "date") if c in frame.columns]
        if not candidates:
            raise ValueError("frame has no recognized temporal coverage column")
        timestamps = pd.to_datetime(frame[candidates[0]], utc=True, errors="coerce").dropna()
        if timestamps.empty:
            raise ValueError("frame has no valid coverage timestamps")
        start, end = timestamps.min(), timestamps.max()
    manifest = SourceManifest(source_id, source_type, pd.Timestamp(start).isoformat(), pd.Timestamp(end).isoformat(), len(frame), dataframe_sha256(frame), retrieval_version, source_uri, schema_version, availability_policy)
    manifest.validate()
    return manifest


def audit_event_timing(events: pd.DataFrame, market_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Audit event availability against every actual market decision timestamp."""
    validate_event_frame(events)
    market = pd.DatetimeIndex(market_index)
    if market.tz is None:
        raise ValueError("market index must be timezone-aware")
    assert_market_session_audit(market)
    rows = []
    for _, event in events.iterrows():
        available = pd.Timestamp(event["available_at"])
        if available.tzinfo is None:
            raise ValueError("event available_at must be timezone-aware")
        available = available.tz_convert("UTC")
        eligible = market[market >= available]
        rows.append({
            "event_id": event["event_id"],
            "available_at": available,
            "first_eligible_decision": eligible[0] if len(eligible) else pd.NaT,
            "published_at": event["published_at"],
            "event_time": event["event_time"],
        })
    return pd.DataFrame(rows)


def assemble_common_observations(market: pd.DataFrame, macro: pd.DataFrame | None = None, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the single observation index from which A/B/C must derive predictions."""
    validate_market_frame(market)
    assert_market_session_audit(market.index)
    base = market.sort_index().copy()
    if macro is not None:
        validate_macro_frame(macro)
        macro_copy = macro.copy()
        if isinstance(macro_copy.index, pd.DatetimeIndex):
            macro_copy = macro_copy[~macro_copy.index.duplicated(keep="last")]
            base = base.join(macro_copy, how="left")
        elif "decision_time" in macro_copy.columns:
            macro_copy["decision_time"] = pd.to_datetime(macro_copy["decision_time"], utc=True)
            macro_copy = macro_copy.set_index("decision_time")
            base = base.join(macro_copy, how="left")
        else:
            raise ValueError("macro must be indexed by decision_time or contain it")
    if events is not None:
        validate_event_frame(events)
        events = deduplicate_events(events)
        audit_event_timing(events, base.index)
        base.attrs["event_count"] = len(events)
    base.attrs["observation_index_hash"] = hashlib.sha256("\n".join(pd.Timestamp(x).isoformat() for x in base.index).encode()).hexdigest()
    return base


def historical_information_set(events: pd.DataFrame, decision_time: str | pd.Timestamp) -> pd.DataFrame:
    """Return exactly the event information available at one historical decision."""
    return admitted_events(deduplicate_events(events), decision_time)


def write_source_manifest(manifests: Iterable[SourceManifest], path: str) -> None:
    items = list(manifests)
    for item in items:
        item.validate()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(item) for item in items], indent=2, sort_keys=True) + "\n", encoding="utf-8")
