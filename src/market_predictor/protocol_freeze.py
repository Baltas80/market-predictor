"""Cryptographic freeze record for the final research protocol."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .reproducibility import canonical_json_hash


@dataclass(frozen=True)
class ProtocolFreeze:
    """Immutable identity of the configuration admitted to final OOS."""

    protocol_fingerprint: str
    dataset_hash: str
    prediction_index_hash: str
    code_version: str
    protocol_version: str

    def validate(self) -> None:
        if not all((self.protocol_fingerprint, self.dataset_hash, self.prediction_index_hash, self.code_version, self.protocol_version)):
            raise ValueError("protocol freeze requires complete identities")

    def fingerprint(self) -> str:
        self.validate()
        return canonical_json_hash(asdict(self))


def freeze_protocol(*, protocol_payload: Any, dataset_hash: str, prediction_index_hash: str, code_version: str, protocol_version: str) -> ProtocolFreeze:
    """Create a deterministic freeze identity before final OOS evaluation."""
    if not dataset_hash or not prediction_index_hash or not code_version or not protocol_version:
        raise ValueError("complete protocol identities are required")
    return ProtocolFreeze(
        protocol_fingerprint=canonical_json_hash(protocol_payload),
        dataset_hash=dataset_hash,
        prediction_index_hash=prediction_index_hash,
        code_version=code_version,
        protocol_version=protocol_version,
    )


def assert_protocol_unchanged(*, frozen: ProtocolFreeze, current_protocol_payload: Any, dataset_hash: str, prediction_index_hash: str, code_version: str, protocol_version: str) -> None:
    """Fail closed if any lockbox-defining identity changed after freezing."""
    frozen.validate()
    current = freeze_protocol(
        protocol_payload=current_protocol_payload,
        dataset_hash=dataset_hash,
        prediction_index_hash=prediction_index_hash,
        code_version=code_version,
        protocol_version=protocol_version,
    )
    if current != frozen:
        raise RuntimeError("final OOS protocol identity changed after freeze; lockbox is invalid")
