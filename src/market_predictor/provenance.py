"""Deterministic provenance records for historical research inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Hash schema, index and values without depending on row order."""
    ordered = frame.sort_index() if isinstance(frame.index, pd.DatetimeIndex) else frame.sort_values(list(frame.columns)).reset_index(drop=True)
    payload = {
        "columns": [str(column) for column in ordered.columns],
        "dtypes": [str(dtype) for dtype in ordered.dtypes],
        "index": [str(value) for value in ordered.index],
        "values": ordered.astype(object).where(pd.notna(ordered), None).to_dict(orient="records"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def source_record(path: str | Path, *, source_id: str, retrieval_method: str, retrieved_at: str) -> dict[str, object]:
    """Create immutable file-level provenance metadata."""
    target = Path(path)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {
        "source_id": source_id,
        "path": str(target),
        "sha256": digest,
        "bytes": target.stat().st_size,
        "retrieval_method": retrieval_method,
        "retrieved_at": retrieved_at,
    }


def assert_unique_keys(frame: pd.DataFrame, keys: list[str], *, name: str) -> None:
    """Fail closed on duplicate source identities."""
    missing = set(keys) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing key columns: {sorted(missing)}")
    if frame.duplicated(keys).any():
        duplicates = int(frame.duplicated(keys).sum())
        raise ValueError(f"{name} contains {duplicates} duplicate rows for keys {keys}")
