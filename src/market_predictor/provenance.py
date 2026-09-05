"""Provenance contracts for historical research inputs."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DatasetProvenance:
    dataset_name: str
    source: str
    retrieved_at: datetime
    content_hash: str
    schema_version: str

    def validate(self) -> None:
        if not self.dataset_name.strip():
            raise ValueError("dataset_name is required")
        if not self.source.strip():
            raise ValueError("source is required")
        if self.retrieved_at.tzinfo is None or self.retrieved_at.utcoffset() is None:
            raise ValueError("retrieved_at must be timezone-aware")
        if len(self.content_hash) != 64:
            raise ValueError("content_hash must be a SHA-256 hexadecimal digest")
        try:
            int(self.content_hash, 16)
        except ValueError as exc:
            raise ValueError("content_hash must be hexadecimal") from exc
        if not self.schema_version.strip():
            raise ValueError("schema_version is required")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
