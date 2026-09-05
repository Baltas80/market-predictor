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


def run_baseline(df: pd.DataFrame, horizon: int = 5, initial_train_fraction: float = 0.6, test_fraction: float = 0.1) -> tuple[pd.DataFrame, list]:
    """Run expanding-window out-of-sample evaluation with a purge gap."""
    if not 0.5 <= initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must be >= 0.5 and < 1")
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be > 0 and < 0.5")
    data = prepare_baseline_data(df, horizon=horizon)
    initial_train_size = max(1, int(len(data) * initial_train_fraction))
    test_size = max(1, int(len(data) * test_fraction))
    from .backtest import make_walk_forward_folds
    folds = make_walk_forward_folds(len(data), initial_train_size=initial_train_size, test_size=test_size, purge=horizon)
    return walk_forward_classification(data, FEATURE_COLUMNS, "target", folds)


def run_final_lockbox(df: pd.DataFrame, horizon: int = 5, test_fraction: float = 0.2) -> tuple[pd.DataFrame, list]:
    """Evaluate one untouched chronological final OOS holdout."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be > 0 and < 0.5")
    data = prepare_baseline_data(df, horizon=horizon)
    assert_market_session_audit(data.index)
    n_rows = len(data)
    test_size = max(1, int(n_rows * test_fraction))
    train_end = n_rows - test_size - horizon
    if train_end < 2:
        raise ValueError("not enough observations for lockbox train, purge, and test")
    fold = Fold(0, train_end, train_end + horizon, n_rows)
    return walk_forward_classification(data, FEATURE_COLUMNS, "target", [fold])


def _shared_final_lockbox_folds(data: pd.DataFrame, horizon: int, test_fraction: float) -> tuple[Fold, ...]:
    """Generate the final A/B/C fold sequence exactly once."""
    n_rows = len(data)
    test_size = max(1, int(n_rows * test_fraction))
    initial_train_size = n_rows - test_size - horizon
    if initial_train_size < 2:
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
    """Run A/B/C through one fail-closed gate and one immutable execution plan."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be > 0 and < 0.5")
    data = prepare_baseline_data(df, horizon=horizon)
    macro_features = macro_features or []
    geopolitical_features = geopolitical_features or []
    required = macro_features + geopolitical_features
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    assert_market_session_audit(data.index)
    gate = ResearchGateState(data_audited=True, information_set_audited=True, abc_protocol_frozen=True)
    gate.assert_can_generate_predictions()
    shared_folds = _shared_final_lockbox_folds(data, horizon, test_fraction)
    plan = _build_final_abc_plan(data, shared_folds, horizon, transaction_cost_bps, slippage_bps)
    results = execute_abc(data, plan=plan, macro_features=macro_features, geopolitical_features=geopolitical_features)
    if any(not result.predictions.index.is_monotonic_increasing for result in results):
        raise ValueError("A/B/C predictions must remain chronological")
    return results


def run_final_lockbox_event_experiments(
    df: pd.DataFrame,
    events: list[MarketEvent],
    *,
    horizon: int = 5,
    test_fraction: float = 0.2,
    macro_features: list[str] | None = None,
    event_features: list[str] | None = None,
    event_half_life_days: float = 7.0,
) -> list[ExperimentResult]:
    """Run A/B/C using timestamped events admitted at their information cutoff."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex for event features")
    event_data = events_to_features(df.index, events, half_life_days=event_half_life_days)
    enriched = df.join(event_data, how="left")
    macro_features = macro_features or []
    if event_features is None:
        event_features = list(event_data.columns)
    missing = [column for column in event_features if column not in enriched.columns]
    if missing:
        raise ValueError(f"Missing event columns: {missing}")
    return run_final_lockbox_experiments(
        enriched, horizon=horizon, test_fraction=test_fraction,
        macro_features=macro_features, geopolitical_features=event_features,
    )


def run_final_lockbox_financial_comparison(
    df: pd.DataFrame,
    *,
    horizon: int = 5,
    test_fraction: float = 0.2,
    macro_features: list[str] | None = None,
    geopolitical_features: list[str] | None = None,
    threshold: float = 0.5,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 0.0,
    periods_per_year: int = 252,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compare A/B/C financial performance on the exact same OOS lockbox."""
    data = prepare_baseline_data(df, horizon=horizon)
    experiments = run_final_lockbox_experiments(
        df, horizon=horizon, test_fraction=test_fraction,
        macro_features=macro_features, geopolitical_features=geopolitical_features,
        transaction_cost_bps=transaction_cost_bps, slippage_bps=slippage_bps,
    )
    rows: list[dict[str, float | str]] = []
    backtests: dict[str, pd.DataFrame] = {}
    reference_index: pd.Index | None = None
    for experiment in experiments:
        frame = experiment.predictions.join(data[["close"]], how="left")
        if reference_index is None:
            reference_index = frame.index
        elif not frame.index.equals(reference_index):
            raise ValueError("A/B/C predictions do not share the same lockbox index")
        backtest_frame, metrics = backtest_long_only(frame, threshold=threshold, transaction_cost_bps=transaction_cost_bps, slippage_bps=slippage_bps, periods_per_year=periods_per_year)
        backtests[experiment.name] = backtest_frame
        rows.append({"experiment": experiment.name, **metrics})
    return pd.DataFrame(rows), backtests
