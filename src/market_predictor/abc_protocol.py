"""Canonical protocol contract for experiments A, B and C."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping, Sequence
import hashlib

import pandas as pd

from .backtest import Fold


@dataclass(frozen=True)
class ABCProtocol:
    """Single source of truth for A/B/C folds, data and execution assumptions."""

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
    folds_hash: str = ""

    def validate(self) -> None:
        if self.lockbox_end < self.lockbox_start:
            raise ValueError("lockbox_end must not precede lockbox_start")
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
        if not self.protocol_version:
            raise ValueError("protocol_version is required")

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
            self.folds_hash,
        )

    def fingerprint(self) -> str:
        """Return a deterministic identity for the complete shared contract."""
        payload = repr(self.compatibility_key()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def folds_identity_hash(folds: Sequence[Fold]) -> str:
    """Hash every fold boundary, not merely its OOS prediction index."""
    payload = [
        {
            "train_start": int(fold.train_start),
            "train_end": int(fold.train_end),
            "test_start": int(fold.test_start),
            "test_end": int(fold.test_end),
        }
        for fold in folds
    ]
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


# Compatibility name retained for callers during the migration. It is an alias,
# not a second implementation or source of truth.
SharedExperimentProtocol = ABCProtocol


def assert_same_abc_protocol(protocols: Mapping[str, ABCProtocol]) -> None:
    """Fail closed unless exactly A/B/C share one immutable protocol identity."""
    required = {"A", "B", "C"}
    if set(protocols) != required:
        raise ValueError("protocol map must contain exactly A, B and C")
    keys = {name: protocol.compatibility_key() for name, protocol in protocols.items()}
    reference = next(iter(keys.values()))
    if any(key != reference for key in keys.values()):
        raise ValueError("A/B/C must use identical observations, folds, purge, lockbox, costs and versions")


def assert_common_prediction_index(predictions: Mapping[str, pd.DataFrame]) -> None:
    if set(predictions) != {"A", "B", "C"}:
        raise ValueError("prediction map must contain exactly A, B and C")
    indices = [predictions[name].index for name in ("A", "B", "C")]
    if not (indices[0].equals(indices[1]) and indices[0].equals(indices[2])):
        raise ValueError("A/B/C predictions must have the identical OOS observation index")


def common_walk_forward_folds(*, n_rows: int, initial_train_size: int, test_size: int, horizon: int):
    """Generate the immutable fold sequence once for the complete A/B/C family."""
    from .backtest import make_walk_forward_folds

    if horizon < 1:
        raise ValueError("horizon must be positive")
    folds = tuple(
        make_walk_forward_folds(
            n_rows,
            initial_train_size=initial_train_size,
            test_size=test_size,
            purge=horizon,
        )
    )
    if not folds:
        raise ValueError("at least one fold is required")
    return folds
