"""Deterministic final-OOS manifest; governance only, never model selection."""
from __future__ import annotations
from dataclasses import asdict,dataclass
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
    result_hashes: tuple[tuple[str,str],...]
    def validate(self):
        if self.oos_end<self.oos_start or self.purge_gap<0: raise ValueError("invalid lockbox dates or purge gap")
        if not self.dataset_hash or not self.code_version or not self.protocol_version: raise ValueError("dataset, code and protocol versions are required")
        if self.transaction_cost_bps<0 or self.slippage_bps<0: raise ValueError("execution costs cannot be negative")
        if not self.result_hashes: raise ValueError("result hashes are required")
    def as_dict(self)->dict[str,Any]:
        self.validate(); payload=asdict(self); payload["oos_start"]=self.oos_start.isoformat(); payload["oos_end"]=self.oos_end.isoformat(); payload["result_hashes"]=list(self.result_hashes); return payload
    def fingerprint(self)->str: return canonical_json_hash(self.as_dict())
