"""Reproducibility helpers for research runs."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_hash(payload: Any) -> str:
    """Hash JSON-compatible configuration deterministically."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_fingerprint(*, dataset_hash: str, config: dict[str, Any], code_version: str) -> str:
    """Create a stable fingerprint from dataset, configuration and code version."""
    if not dataset_hash or not code_version:
        raise ValueError("dataset_hash and code_version are required")
    return canonical_json_hash({"dataset_hash": dataset_hash, "config": config, "code_version": code_version})
