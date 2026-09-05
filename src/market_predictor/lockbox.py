"""Governance primitives for an immutable final OOS lockbox."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import Any

from .reproducibility import canonical_json_hash, run_fingerprint


@dataclass(frozen=True)
class LockboxSpec:
    start: date
    end: date
    purge_gap: int
    dataset_hash: str
    code_version: str
    protocol_version: str
    transaction_cost_bps: float
    slippage_bps: float

    def validate(self) -> None:
        if self.end < self.start:
            raise ValueError("lockbox end must not precede start")
        if self.purge_gap < 0:
            raise ValueError("purge_gap cannot be negative")
        if not self.dataset_hash or not self.code_version or not self.protocol_version:
            raise ValueError("dataset, code and protocol versions are required")
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs cannot be negative")

    def fingerprint(self, *, model_configs: dict[str, Any]) -> str:
        self.validate()
        payload = {"lockbox": asdict(self), "model_configs": model_configs}
        payload["lockbox"]["start"] = self.start.isoformat()
        payload["lockbox"]["end"] = self.end.isoformat()
        return run_fingerprint(dataset_hash=self.dataset_hash, config=payload, code_version=self.code_version)


def result_artifact_hash(result: Any) -> str:
    return canonical_json_hash(result)
