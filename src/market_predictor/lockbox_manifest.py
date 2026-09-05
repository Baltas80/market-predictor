"""Immutable manifest contract for final out-of-sample research artifacts."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any
from .reproducibility import canonical_json_hash

@dataclass(frozen=True)
class LockboxManifest:
    oos_start: date
    oos_end: date
    purge_gap: int
    dataset_hash: str
    code_version: str
    protocol_version: str
    transaction_cost_bps: float
    slippage_bps: float
    result_hashes: tuple[tuple[str, str], ...] = ()

    def validate(self) -> None:
        if self.oos_end < self.oos_start: raise ValueError("oos_end must not precede oos_start")
        if self.purge_gap < 0: raise ValueError("purge_gap cannot be negative")
        if not self.dataset_hash or not self.code_version or not self.protocol_version: raise ValueError("dataset, code and protocol versions are required")
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0: raise ValueError("execution costs cannot be negative")
        names = [name for name, _ in self.result_hashes]
        if len(names) != len(set(names)): raise ValueError("result artifact names must be unique")

    def as_payload(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["oos_start"] = self.oos_start.isoformat()
        payload["oos_end"] = self.oos_end.isoformat()
        payload["result_hashes"] = list(self.result_hashes)
        return payload

    def manifest_hash(self) -> str:
        return canonical_json_hash(self.as_payload())
