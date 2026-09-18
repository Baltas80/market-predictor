"""Deterministic final-OOS manifest; governance only, never model selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

from .reproducibility import canonical_json_hash


@dataclass(frozen=True)
class LockboxManifest:
    """Complete provenance record for one immutable final-OOS evaluation."""

    experiment_id: str
    git_commit: str
    branch: str
    dataset_hash: str
    source_hashes: tuple[tuple[str, str], ...]
    coverage: dict[str, Any]
    feature_hash: str
    model: str
    hyperparameters: dict[str, Any]
    seeds: tuple[int, ...]
    fold_definition: tuple[dict[str, Any], ...]
    train_ranges: tuple[tuple[str, str], ...]
    validation_ranges: tuple[tuple[str, str], ...]
    oos_ranges: tuple[tuple[str, str], ...]
    purge_gap: int
    embargo: int
    transaction_cost_bps: float
    slippage_bps: float
    benchmark: str
    software_version: str
    python_version: str
    dependencies: tuple[str, ...]
    result_hashes: tuple[tuple[str, str], ...]
    artifact_paths: tuple[str, ...]
    execution_timestamp: str
    oos_start: date
    oos_end: date
    protocol_version: str
    observations: int

    def validate(self) -> None:
        if self.oos_end < self.oos_start or self.purge_gap < 0 or self.embargo < 0:
            raise ValueError("invalid lockbox dates, purge gap or embargo")
        required_strings = {
            "experiment_id": self.experiment_id,
            "git_commit": self.git_commit,
            "branch": self.branch,
            "dataset_hash": self.dataset_hash,
            "feature_hash": self.feature_hash,
            "model": self.model,
            "benchmark": self.benchmark,
            "software_version": self.software_version,
            "python_version": self.python_version,
            "execution_timestamp": self.execution_timestamp,
            "protocol_version": self.protocol_version,
        }
        if any(not value for value in required_strings.values()):
            raise ValueError("complete lockbox provenance fields are required")
        if len(self.git_commit) != 40:
            raise ValueError("git_commit must be a full 40-character SHA")
        if not self.source_hashes:
            raise ValueError("source hashes are required")
        if not self.fold_definition:
            raise ValueError("fold definition is required")
        if not self.train_ranges or not self.oos_ranges:
            raise ValueError("train and OOS ranges are required")
        if self.observations <= 0:
            raise ValueError("observations must be positive")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("execution costs cannot be negative")
        if not self.dependencies:
            raise ValueError("dependency provenance is required")
        if not self.result_hashes:
            raise ValueError("result hashes are required")
        if not self.artifact_paths:
            raise ValueError("artifact paths are required")
        try:
            datetime.fromisoformat(self.execution_timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("execution_timestamp must be ISO-8601") from exc

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["oos_start"] = self.oos_start.isoformat()
        payload["oos_end"] = self.oos_end.isoformat()
        payload["source_hashes"] = list(self.source_hashes)
        payload["fold_definition"] = list(self.fold_definition)
        payload["train_ranges"] = list(self.train_ranges)
        payload["validation_ranges"] = list(self.validation_ranges)
        payload["oos_ranges"] = list(self.oos_ranges)
        payload["seeds"] = list(self.seeds)
        payload["dependencies"] = list(self.dependencies)
        payload["result_hashes"] = list(self.result_hashes)
        payload["artifact_paths"] = list(self.artifact_paths)
        return payload

    def fingerprint(self) -> str:
        """Return a deterministic identity for the complete immutable manifest."""
        return canonical_json_hash(self.as_dict())
