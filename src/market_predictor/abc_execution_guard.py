"""Fail-closed execution guard for the A/B/C experiment family."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from .abc_protocol import ABCProtocol, assert_same_abc_protocol
from .backtest import Fold
from .experiments import ExperimentResult, run_feature_ablation
from .reproducibility import canonical_json_hash


@dataclass(frozen=True)
class ABCExecutionPlan:
    """Immutable inputs shared by every A/B/C execution."""

    protocol: ABCProtocol
    folds: tuple[Fold, ...]
    observation_index: pd.Index
    target: str = "target"

    def validate(self) -> None:
        self.protocol.validate()
        if not self.folds:
            raise ValueError("A/B/C execution requires at least one shared fold")
        if not self.observation_index.is_monotonic_increasing:
            raise ValueError("shared observation index must be chronological")
        if not self.observation_index.is_unique:
            raise ValueError("shared observation index must be unique")
        if not self.target:
            raise ValueError("target is required")
        for fold in self.folds:
            if fold.train_start < 0 or fold.train_end <= fold.train_start:
                raise ValueError("invalid shared training fold")
            if fold.test_start < fold.train_end or fold.test_end <= fold.test_start:
                raise ValueError("invalid shared test fold")
            if fold.test_start - fold.train_end < self.protocol.purge_gap:
                raise ValueError("shared fold violates the protocol purge gap")
            if fold.test_end > len(self.observation_index):
                raise ValueError("shared fold exceeds the observation index")


def _prediction_index_for_folds(observation_index: pd.Index, folds: Sequence[Fold]) -> pd.Index:
    """Return exactly the OOS observations implied by the shared fold sequence."""
    pieces = [observation_index[fold.test_start:fold.test_end] for fold in folds]
    if not pieces:
        return observation_index[:0]
    return pieces[0].append(pieces[1:])


def _prediction_index_hash(index: pd.Index) -> str:
    return canonical_json_hash([timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp) for timestamp in index])


def build_abc_execution_plan(*, protocol: ABCProtocol, folds: Sequence[Fold], observation_index: pd.Index, target: str = "target") -> ABCExecutionPlan:
    """Construct and validate the only execution plan accepted by A/B/C."""
    plan = ABCExecutionPlan(protocol, tuple(folds), observation_index.copy(), target)
    plan.validate()
    if len(plan.observation_index) != protocol.observations:
        raise ValueError("protocol observations do not match the shared observation index")
    expected_index = _prediction_index_for_folds(plan.observation_index, plan.folds)
    if not expected_index.is_unique:
        raise ValueError("shared folds must produce a unique OOS prediction index")
    expected_hash = _prediction_index_hash(expected_index)
    if protocol.prediction_index_hash != expected_hash:
        raise ValueError("protocol prediction_index_hash does not match the shared fold OOS index")
    return plan


def execute_abc(data: pd.DataFrame, *, plan: ABCExecutionPlan, macro_features: Sequence[str] = (), geopolitical_features: Sequence[str] = ()) -> list[ExperimentResult]:
    """Execute A/B/C from one immutable plan; feature groups are the only difference."""
    plan.validate()
    if not data.index.equals(plan.observation_index):
        raise ValueError("A/B/C data must use the exact shared observation index")
    if plan.target not in data.columns:
        raise ValueError(f"missing target column: {plan.target}")
    expected_index = _prediction_index_for_folds(plan.observation_index, plan.folds)
    expected_hash = _prediction_index_hash(expected_index)
    if plan.protocol.prediction_index_hash != expected_hash:
        raise ValueError("protocol prediction_index_hash does not match the execution plan")
    results = run_feature_ablation(
        data,
        list(plan.folds),
        macro_features=list(macro_features),
        geopolitical_features=list(geopolitical_features),
        target=plan.target,
    )
    names = tuple(result.name for result in results)
    if names != ("technical", "technical_macro", "technical_macro_geopolitical"):
        raise AssertionError("A/B/C execution returned an unexpected experiment family")
    if not results:
        raise ValueError("A/B/C execution returned no experiments")
    reference_index = results[0].predictions.index
    if not reference_index.equals(expected_index):
        raise ValueError("A/B/C predictions do not match the shared fold OOS observations")
    if _prediction_index_hash(reference_index) != plan.protocol.prediction_index_hash:
        raise ValueError("A/B/C prediction index hash does not match the frozen protocol")
    for result in results[1:]:
        if not result.predictions.index.equals(reference_index):
            raise ValueError("A/B/C predictions diverged in OOS observations")
        if _prediction_index_hash(result.predictions.index) != plan.protocol.prediction_index_hash:
            raise ValueError("A/B/C prediction index hash diverged from the frozen protocol")
    return results


def validate_abc_protocol_family(protocols: Mapping[str, ABCProtocol]) -> None:
    """Public fail-closed check used before comparison or financial evaluation."""
    assert_same_abc_protocol(protocols)
