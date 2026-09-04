"""Deterministic event deduplication utilities."""

from __future__ import annotations

import pandas as pd


def deduplicate_events(events: pd.DataFrame, id_column: str = "global_event_id") -> pd.DataFrame:
    """Keep one deterministic record per source event identifier."""
    if id_column not in events.columns:
        raise ValueError(f"Missing event identifier column: {id_column}")
    out = events.copy()
    out = out.dropna(subset=[id_column])
    out = out.drop_duplicates(subset=[id_column], keep="first")
    sort_columns = [column for column in ("event_time", "available_time", id_column) if column in out.columns]
    if sort_columns:
        out = out.sort_values(sort_columns, kind="mergesort")
    return out.reset_index(drop=True)
