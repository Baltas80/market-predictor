"""Market-data ingestion from a normalized OHLCV CSV file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_OHLCV = ("open", "high", "low", "close", "volume")


def load_ohlcv_csv(path: str | Path, *, timestamp_column: str = "timestamp") -> pd.DataFrame:
    """Load and validate chronological OHLCV data.

    The adapter deliberately accepts local CSV rather than coupling the model
    to a vendor. Raw downloads should be stored separately with provenance.
    """
    frame = pd.read_csv(path)
    missing = set(REQUIRED_OHLCV) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if timestamp_column not in frame.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")

    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    if frame[timestamp_column].isna().any():
        raise ValueError("timestamp contains invalid values")

    for column in REQUIRED_OHLCV:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(REQUIRED_OHLCV)].isna().any().any():
        raise ValueError("OHLCV contains non-numeric or missing values")

    frame = frame.sort_values(timestamp_column).drop_duplicates(timestamp_column, keep="last")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise ValueError("high is below open/close")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise ValueError("low is above open/close")
    if (frame["volume"] < 0).any():
        raise ValueError("volume cannot be negative")

    return frame.set_index(timestamp_column)
