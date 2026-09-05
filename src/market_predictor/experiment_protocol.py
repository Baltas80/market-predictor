"""Shared protocol contract for A/B/C final OOS experiments.

The contract is deliberately model-agnostic: every variant must receive the
same observations, folds, target horizon, purge gap, lockbox identity and
execution assumptions. Variant-specific features are the only intended
experimental difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable


@dataclass(frozen=True)
class SharedExperimentProtocol:
    lockbox_start: date
    lockbox_end: date
    purge_gap: int
    horizon: int
    transaction_cost_bps: float
    slippage_bps: float
    observations: int
    prediction_index_hash: str
    dataset_hash: str
    code_version: str
    protocol_version: str = "abc-v1"

    def validate(self) -> None:
        if self.lockbox_end < self.lockbox_start:
            raise ValueError("lockbox_end must not precede lockbox_start")
        if self.purge_gap < 0:
            raise ValueError("purge_gap must be >= 0")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.purge_gap < self.horizon:
            raise ValueError("purge_gap must cover the target horizon")
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs cannot be negative")
        if self.observations <= 0:
            raise ValueError("observations must be positive")
        if not self.prediction_index_hash or not self.dataset_hash or not self.code_version:
            raise ValueError("dataset, code and prediction index identity are required")

    def compatibility_key(self) -> tuple[object, ...]:
        """Return every immutable setting that A/B/C must share."""
        self.validate()
        return (
            self.lockbox_start,
            self.lockbox_end,
            self.purge_gap,
            self.horizon,
            float(self.transaction_cost_bps),
            float(self.slippage_bps),
            self.observations,
            self.prediction_index_hash,
            self.dataset_hash,
            self.code_version,
            self.protocol_version,
        )


def assert_common_protocol(protocols: dict[str, SharedExperimentProtocol]) -> None:
    """Reject any A/B/C protocol mismatch before OOS results are compared."""
    if not protocols:
        raise ValueError("at least one experiment protocol is required")
    keys = {name: protocol.compatibility_key() for name, protocol in protocols.items()}
    reference_name, reference = next(iter(keys.items()))
    mismatches = [name for name, key in keys.items() if key != reference]
    if mismatches:
        raise ValueError(
            f"experiments {mismatches} do not share the protocol of {reference_name}"
        )


def assert_oos_configuration_frozen(*, configuration_fingerprint: str, recorded_fingerprint: str) -> None:
    """Require a pre-recorded configuration identity before final OOS use."""
    if not configuration_fingerprint or not recorded_fingerprint:
        raise ValueError("configuration fingerprints are required")
    if configuration_fingerprint != recorded_fingerprint:
        raise ValueError("final OOS configuration differs from the recorded protocol")
