"""GDELT 2.0 event normalization helpers."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

import pandas as pd

GDELT_COLUMNS = {
    "GlobalEventID": "global_event_id", "SQLDATE": "event_date", "DATEADDED": "available_time",
    "EventCode": "cameo_code", "QuadClass": "quad_class", "GoldsteinScale": "goldstein_scale",
    "NumMentions": "num_mentions", "NumSources": "num_sources", "AvgTone": "avg_tone", "SOURCEURL": "source_url",
}


def _canonical_date_added(value: object) -> str | pd._libs.missing.NAType:
    """Canonicalize exact integer-like GDELT DATEADDED encodings only."""
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text.isdigit() and len(text) == 14:
        return text
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return pd.NA
    if not number.is_finite() or number != number.to_integral_value():
        return pd.NA
    canonical = format(number.to_integral_value(), "f")
    return canonical if canonical.isdigit() and len(canonical) == 14 else pd.NA


def normalize_gdelt_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize selected GDELT 2.0 event columns to the project schema.

    ``SOURCEURL`` is optional because some reduced exports/tests omit it; when
    absent, the normalized field is retained with a missing value.
    """
    required = set(GDELT_COLUMNS) - {"SOURCEURL"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing GDELT columns: {sorted(missing)}")

    columns = list(required)
    if "SOURCEURL" in raw.columns:
        columns.append("SOURCEURL")
    out = raw[columns].rename(columns=GDELT_COLUMNS).copy()
    if "source_url" not in out.columns:
        out["source_url"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["event_date"] = pd.to_datetime(out["event_date"].astype("string"), format="%Y%m%d", utc=True, errors="coerce")
    canonical = out["available_time"].map(_canonical_date_added).astype("string")
    out["available_time"] = pd.to_datetime(canonical, format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    if out[["event_date", "available_time"]].isna().any().any():
        raise ValueError("GDELT event or availability timestamps are invalid")
    if (out["available_time"] < out["event_date"]).any():
        raise ValueError("GDELT availability cannot precede the event date")
    out["cameo_code"] = out["cameo_code"].astype("string").str.strip()
    for column in ("quad_class", "goldstein_scale", "num_mentions", "num_sources", "avg_tone"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    visibility = 1.0 + out["num_mentions"].clip(lower=0).fillna(0).pow(0.5).clip(upper=1000)
    out["intensity"] = out["goldstein_scale"].abs().fillna(0.0) * visibility
    out["event_type"] = out["quad_class"].map({1: "verbal_cooperation", 2: "material_cooperation", 3: "verbal_conflict", 4: "material_conflict"}).fillna("unknown")
    out["event_time"] = out["event_date"]
    return out[["global_event_id", "event_time", "available_time", "event_type", "intensity", "cameo_code", "quad_class", "goldstein_scale", "num_mentions", "num_sources", "avg_tone", "source_url"]]
