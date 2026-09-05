"""Chronological, purged walk-forward baseline pipeline."""

from __future__ import annotations

import pandas as pd

from .abc_execution_guard import build_abc_execution_plan, execute_abc
from .abc_protocol import ABCProtocol, common_walk_forward_folds
from .backtest import Fold, walk_forward_classification
from .event_features import events_to_features
from .event_schema import MarketEvent
from .experiments import ExperimentResult
from .features import add_market_features, make_target
from .financial import backtest_long_only
from .historical_ingestion import dataframe_sha256
from .reproducibility import canonical_json_hash
from .research_gate import ResearchGateState, assert_market_session_audit
from .time_contract import assert_market_target_contract

FEATURE_COLUMNS = [
    "return_1d", "return_5d", "volatility_20d", "price_to_sma20", "volume_change", "range_pct",
]
PROTOCOL_VERSION = "2026-09-05"


def prepare_baseline_data(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Build model-ready features and remove rows whose target is unknown."""
    data = add_market_features(df)
    assert_market_target_contract(data.index, horizon)
    data["target"] = make_target(data, horizon=horizon)
    return data.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()


def run_baseline(df: pd.DataFrame, horizon: int = 5):
    data = prepare_baseline_data(df, horizon=horizon)
    return walk_forward_classification(data, FEATURE_COLUMNS, "target", common_walk_forward_folds(n_rows=len(data), initial_train_size=max(20, len(data) // 2), test_size=max(1, len(data) // 10), horizon=horizon))


def _shared_final_lockbox_folds(data: pd.DataFrame, horizon: int, test_fraction: float) -> tuple[Fold, ...]:
    n_rows = len(data)
    test_size = max(1, int(n_rows * test_fraction))
    initial_train_size = n_rows - test_size - horizon
    if initial_train_size <= 0:
        raise ValueError("not enough observations for lockbox train, purge, and test")
    return tuple(common_walk_forward_folds(n_rows=n_rows, initial_train_size=initial_train_size, test_size=test_size, horizon=horizon))


def _final_lockbox_fold(data: pd.DataFrame, horizon: int, test_fraction: float) -> Fold:
    """Backward-compatible single-fold accessor backed by the shared generator."""
    folds = _shared_final_lockbox_folds(data, horizon, test_fraction)
    if len(folds) != 1:
        raise ValueError("final lockbox must contain exactly one fold")
    return folds[0]


def _build_final_abc_plan(data: pd.DataFrame, folds: tuple[Fold, ...], horizon: int, transaction_cost_bps: float, slippage_bps: float):
    """Create the sole immutable A/B/C execution plan after all input audits."""
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is None:
        raise ValueError("final A/B/C execution requires a timezone-aware market decision index")
    assert_market_session_audit(data.index)
    test_parts = [data.index[fold.test_start:fold.test_end] for fold in folds]
    test_index = test_parts[0].append(test_parts[1:])
    prediction_hash = canonical_json_hash([ts.isoformat() for ts in test_index])
    protocol = ABCProtocol(
        lockbox_start=pd.Timestamp(test_index[0]).date(),
        lockbox_end=pd.Timestamp(test_index[-1]).date(),
        purge_gap=horizon,
        horizon=horizon,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        observations=len(data),
        prediction_index_hash=prediction_hash,
        dataset_hash=dataframe_sha256(data),
        code_version="pipeline.final_abc",
        protocol_version=PROTOCOL_VERSION,
    )
    plan = build_abc_execution_plan(protocol=protocol, folds=folds, observation_index=data.index, target="target")
    return plan


def run_final_lockbox_experiments(
    df: pd.DataFrame,
    *,
    horizon: int = 5,
    test_fraction: float = 0.2,
    macro_features: list[str] | None = None,
    geopolitical_features: list[str] | None = None,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 0.0,
) -> list[ExperimentResult]:
    """Run A/B/C using exactly one shared lockbox fold sequence."""
    data = prepare_baseline_data(df, horizon=horizon)
    shared_folds = _shared_final_lockbox_folds(data, horizon, test_fraction)
    plan = _build_final_abc_plan(data, shared_folds, horizon, transaction_cost_bps, slippage_bps)
    results = execute_abc(
        data,
        plan=plan,
        macro_features=macro_features or [],
        geopolitical_features=geopolitical_features or [],
    )
    return results


def run_final_lockbox_event_experiments(
    df: pd.DataFrame,
    events: list[MarketEvent],
    *,
    event_features: list[str],
    horizon: int = 5,
    test_fraction: float = 0.2,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 0.0,
) -> list[ExperimentResult]:
    """Run the common A/B/C lockbox with event-derived features in C."""
    event_frame = events_to_features(df.index, events)
    data = prepare_baseline_data(df, horizon=horizon).join(event_frame, how="left").fillna(0.0)
    return run_final_lockbox_experiments(
        data,
        horizon=horizon,
        test_fraction=test_fraction,
        macro_features=[],
        geopolitical_features=event_features,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )


def run_final_lockbox_financial_comparison(
    df: pd.DataFrame,
    *,
    horizon: int = 5,
    test_fraction: float = 0.2,
    macro_features: list[str] | None = None,
    geopolitical_features: list[str] | None = None,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 0.0,
    threshold: float = 0.5,
):
    """Evaluate A/B/C predictions under one fixed financial protocol."""
    experiments = run_final_lockbox_experiments(
        df,
        horizon=horizon,
        test_fraction=test_fraction,
        macro_features=macro_features,
        geopolitical_features=geopolitical_features,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    prediction_indices = {result.name: result.predictions.index for result in experiments}
    reference = prediction_indices[experiments[0].name]
    if any(not index.equals(reference) for index in prediction_indices.values()):
        raise ValueError("financial comparison requires identical A/B/C prediction indices")
    backtests = {}
    for result in experiments:
        backtests[result.name] = backtest_long_only(
            result.predictions["probability"],
            df.loc[result.predictions.index, "close"].pct_change().fillna(0.0),
            threshold=threshold,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )
    return experiments, backtests
