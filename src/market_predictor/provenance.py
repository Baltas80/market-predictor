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


def source_record(path: str | Path, *, source_id: str, retrieval_method: str, retrieved_at: str, normalized_frame: pd.DataFrame | None = None, coverage_start: str = "", coverage_end: str = "", version: str = "", pit_policy: str = "") -> dict[str, object]:
    """Create immutable raw/normalized provenance metadata for one source."""
    target = Path(path)
    raw_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    record: dict[str, object] = {
        "source_id": source_id,
        "path": str(target),
        "raw_sha256": raw_hash,
        "sha256": raw_hash,
        "bytes": target.stat().st_size,
        "retrieval_method": retrieval_method,
        "retrieved_at": retrieved_at,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "version": version,
        "pit_policy": pit_policy,
    }
    if normalized_frame is not None:
        record["normalized_sha256"] = dataframe_fingerprint(normalized_frame)
    return record


def observation_trace(*, source_id: str, raw_sha256: str, normalized_sha256: str, observation_id: str, decision_time: str, pit_policy: str, source_version: str = "") -> dict[str, str]:
    """Record the provenance chain from source artifacts to one admitted observation."""
    if not all((source_id, raw_sha256, normalized_sha256, observation_id, decision_time, pit_policy)):
        raise ValueError("complete observation provenance is required")
    return {
        "source_id": source_id,
        "raw_sha256": raw_sha256,
        "normalized_sha256": normalized_sha256,
        "observation_id": observation_id,
        "decision_time": decision_time,
        "pit_policy": pit_policy,
        "source_version": source_version,
    }


def assert_unique_keys(frame: pd.DataFrame, keys: list[str], *, name: str) -> None:
    """Fail closed on duplicate source identities."""
    missing = set(keys) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing key columns: {sorted(missing)}")
    if frame.duplicated(keys).any():
        duplicates = int(frame.duplicated(keys).sum())
        raise ValueError(f"{name} contains {duplicates} duplicate rows for keys {keys}")
