"""Fail-closed admission gate for real historical research datasets."""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .fred_pit import audit_fred_point_in_time
from .historical_ingestion import SourceManifest
from .research_schema import validate_event_frame, validate_macro_frame, validate_market_frame
from .research_gate import assert_market_session_audit


def validate_historical_dataset(
    market: pd.DataFrame,
    *,
    macro: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    manifests: Iterable[SourceManifest] = (),
) -> dict[str, object]:
    """Validate all admissible historical inputs before model execution."""
    validate_market_frame(market)
    session_audit = assert_market_session_audit(market.index)
    if macro is not None:
        validate_macro_frame(macro)
        if {"series_id", "observation_date", "value", "vintage_start", "vintage_end"}.issubset(macro.columns):
            fred = macro.rename(columns={"observation_date": "date", "vintage_start": "realtime_start", "vintage_end": "realtime_end"})
            audit_fred_point_in_time(fred[["series_id", "date", "value", "realtime_start", "realtime_end"]])
    if events is not None:
        validate_event_frame(events)
    manifest_items = tuple(manifests)
    for manifest in manifest_items:
        manifest.validate()
    if not manifest_items:
        raise RuntimeError("historical gate requires at least one source manifest")
    return {
        "status": "admissible",
        "market_observations": len(market),
        "macro_rows": 0 if macro is None else len(macro),
        "event_rows": 0 if events is None else len(events),
        "manifest_count": len(manifest_items),
        "session_rows": len(session_audit),
    }
