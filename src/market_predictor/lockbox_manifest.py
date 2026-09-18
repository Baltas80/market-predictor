"""Deterministic final-OOS manifest; governance only, never model selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import re
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
    result_hashes: tuple[tuple[str, str], ...]
    experiment_id: str = ""
    git_commit: str = ""
    branch: str = ""
    source_hashes: tuple[tuple[str, str], ...] = ()
    coverage: tuple[tuple[str, Any], ...] = ()
    feature_hash: str = ""
    model: str = ""
    hyperparameters: tuple[tuple[str, Any], ...] = ()
    seeds: tuple[int, ...] = ()
    fold_definition: str = ""
    folds_hash: str = ""
    train_ranges: tuple[tuple[str, str], ...] = ()
    validation_ranges: tuple[tuple[str, str], ...] = ()
    oos_ranges: tuple[tuple[str, str], ...] = ()
    embargo: int = 0
    benchmark: str = ""
    software_version: str = ""
    python_version: str = ""
    dependencies: tuple[tuple[str, str], ...] = ()
    artifact_paths: tuple[str, ...] = ()
    execution_timestamp: str = ""

    def validate(self):
        if self.oos_end < self.oos_start or self.purge_gap < 0 or self.embargo < 0:
            raise ValueError("invalid lockbox dates, purge gap or embargo")
        if not self.dataset_hash or not self.code_version or not self.protocol_version:
            raise ValueError("dataset, code and protocol versions are required")
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("execution costs cannot be negative")
        if not self.result_hashes:
            raise ValueError("result hashes are required")
        if self.experiment_id:
            if not re.fullmatch(r"[0-9a-f]{40}", self.git_commit):
                raise ValueError("git_commit must be a 40-character Git SHA for a final experiment")
            required = {
                "branch": self.branch,
                "source_hashes": self.source_hashes,
                "feature_hash": self.feature_hash,
                "model": self.model,
                "fold_definition": self.fold_definition,
                "folds_hash": self.folds_hash,
                "benchmark": self.benchmark,
                "python_version": self.python_version,
                "execution_timestamp": self.execution_timestamp,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(f"required final experiment provenance missing: {missing}")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["oos_start"] = self.oos_start.isoformat()
        payload["oos_end"] = self.oos_end.isoformat()
        for field in (
            "result_hashes", "source_hashes", "coverage", "hyperparameters",
            "train_ranges", "validation_ranges", "oos_ranges", "dependencies",
        ):
            payload[field] = [list(item) for item in getattr(self, field)]
        payload["seeds"] = list(self.seeds)
        payload["artifact_paths"] = list(self.artifact_paths)
        return payload

    def fingerprint(self) -> str:
        return canonical_json_hash(self.as_dict())
