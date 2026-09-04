"""GDELT event normalization helpers.

The network/download layer is intentionally kept outside this module so raw
GDELT files can be archived and checksummed before normalization.
"""

from __future__ import annotations

import pandas as pd

GDELT_COLUMNS = {
    "GlobalEventID": "global_event_id",
    "SQLDATE": "event_date",
    "DATEADDED": "available_time",
    "EventCode": "cameo_code",
    "QuadClass": "quad_class",
    "GoldsteinScale": "goldstein_scale",
    "NumMentions": "num_mentions",
    "NumSources": "num_sources",
    "AvgTone": "avg_tone",
}


def normalize_gdelt_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize selected GDELT 2.0 event columns to the project schema."""
    missing = set(GDELT_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing GDELT columns: {sorted(missing)}")

    out = raw[list(GDELT_COLUMNS)].rename(columns=GDELT_COLUMNS).copy()
    out["event_date"] = pd.to_datetime(out["event_date"].astype(str), format="%Y%m%d", utc=True, errors="coerce")
    out["available_time"] = pd.to_datetime(
        out["available_time"].astype(str), format="%Y%m%d%H%M%S", utc=True, errors="coerce"
    )
    if out[["event_date", "available_time"]].isna().any().any():
        raise ValueError("GDELT event or availability timestamps are invalid")

    for column in ("cameo_code", "quad_class", "goldstein_scale", "num_mentions", "num_sources", "avg_tone"):
        out[column] = pd.to_numeric(out[column], errors="coerce")

    # A reproducible bounded intensity proxy. Goldstein measures event impact;
    # mention/source counts provide a conservative visibility multiplier.
    visibility = (1.0 + out["num_mentions"].clip(lower=0).fillna(0).clip(upper=1000).pow(0.5))
    out["intensity"] = out["goldstein_scale"].abs().fillna(0.0) * visibility
    out["event_type"] = out["quad_class"].map(
        {1: "verbal_cooperation", 2: "material_cooperation", 3: "verbal_conflict", 4: "material_conflict"}
    ).fillna("unknown")
    out["event_time"] = out["event_date"]
    return out[["global_event_id", "event_time", "available_time", "event_type", "intensity", "cameo_code", "quad_class", "goldstein_scale", "num_mentions", "num_sources", "avg_tone"]]
