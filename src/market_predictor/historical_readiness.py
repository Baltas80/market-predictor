"""Pre-ingestion gates for historical datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_historical_frame(
    frame: pd.DataFrame, *, timestamp_column=None, required_columns=()
):
    """Fail closed on missing columns, invalid timestamps, duplicates or non-finite data."""
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"missing required historical columns: {sorted(missing)}")

    index = (
        frame.index
        if timestamp_column is None
        else pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    )
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("historical timestamps must be a DatetimeIndex")
    if index.hasnans:
        raise ValueError("historical timestamps contain invalid/missing values")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("historical timestamps must be unique and chronological")

    numeric = frame.select_dtypes(include="number")
    if not numeric.empty and not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("historical numeric data contains missing/non-finite values")
    return True
