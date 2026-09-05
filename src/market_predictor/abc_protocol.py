"""Shared protocol contract for experiments A, B and C."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import hashlib
import pandas as pd


@dataclass(frozen=True)
class ABCProtocol:
    """Single source of truth for folds, purge, lockbox and execution assumptions."""
    horizon: int
    purge_gap: int
    lockbox_start: str
    lockbox_end: str
    transaction_cost_bps: float
    slippage_bps: float
    dataset_hash: str
    code_version: str
    protocol_version: str

    def validate(self) -> None:
        if self.horizon < 1 or self.purge_gap < self.horizon:
            raise ValueError("purge_gap must be at least the target horizon")
        if pd.Timestamp(self.lockbox_end) < pd.Timestamp(self.lockbox_start):
            raise ValueError("invalid lockbox interval")
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("execution costs cannot be negative")
        if not all((self.dataset_hash, self.code_version, self.protocol_version)):
            raise ValueError("dataset/code/protocol identity is required")

    def fingerprint(self) -> str:
        self.validate()
        payload = "|".join(map(str, (self.horizon, self.purge_gap, self.lockbox_start, self.lockbox_end, self.transaction_cost_bps, self.slippage_bps, self.dataset_hash, self.code_version, self.protocol_version)))
        return hashlib.sha256(payload.encode()).hexdigest()


def assert_same_abc_protocol(protocols: Mapping[str, ABCProtocol]) -> None:
    """Fail closed unless A/B/C have exactly one shared protocol identity."""
    required = ("A", "B", "C")
    if set(protocols) != set(required):
        raise ValueError("protocol map must contain exactly A, B and C")
    fingerprints = {protocol.fingerprint() for protocol in protocols.values()}
    if len(fingerprints) != 1:
        raise ValueError("A/B/C must use identical folds, purge, lockbox, costs and versions")


def assert_common_prediction_index(predictions: Mapping[str, pd.DataFrame]) -> None:
    if set(predictions) != {"A", "B", "C"}:
        raise ValueError("prediction map must contain exactly A, B and C")
    indices = [predictions[name].index for name in ("A", "B", "C")]
    if not (indices[0].equals(indices[1]) and indices[0].equals(indices[2])):
        raise ValueError("A/B/C predictions must have the identical OOS observation index")


def common_walk_forward_folds(*, n_rows: int, initial_train_size: int, test_size: int, horizon: int):
    """Generate folds once; callers must reuse the returned immutable sequence."""
    from .backtest import make_walk_forward_folds
    if horizon < 1:
        raise ValueError("horizon must be positive")
    folds = tuple(make_walk_forward_folds(n_rows, initial_train_size=initial_train_size, test_size=test_size, purge=horizon))
    if not folds:
        raise ValueError("at least one fold is required")
    return folds
