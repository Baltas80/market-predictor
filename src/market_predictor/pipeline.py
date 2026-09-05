"""Chronological, purged walk-forward baseline pipeline."""

from __future__ import annotations

import pandas as pd

from .backtest import Fold, make_walk_forward_folds, walk_forward_classification
from .event_features import events_to_features
from .event_schema import MarketEvent
from .experiments import ExperimentResult, run_feature_ablation
from .features import add_market_features, make_target
from .financial import backtest_long_only
from .time_contract import assert_market_target_contract

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "volatility_20d",
    "price_to_sma20",
    "volume_change",
    "range_pct",
]


def prepare_baseline_data(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Build model-ready features and remove rows whose target is unknown."""
    data = add_market_features(df)
    assert_market_target_contract(data.index, horizon)
    data["target"] = make_target(data, horizon=horizon)
    return data.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()


def run_baseline(
    df: pd.DataFrame,
    horizon: int = 5,
    initial_train_fraction: float = 0.6,
    test_fraction: float = 0.1,
) -> tuple[pd.DataFrame, list]:
    """Run expanding-window out-of-sample evaluation with a purge gap."""
    if not 0.5 <= initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must be >= 0.5 and < 1")
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be > 0 and < 0.5")

    data = prepare_baseline_data(df, horizon=horizon)
    initial_train_size = max(1, int(len(data) * initial_train_fraction))
    test_size = max(1, int(len(data) * test_fraction))
    folds = make_walk_forward_folds(
        len(data),
        initial_train_size=initial_train_size,
        test_size=test_size,
        purge=horizon,
    )
    return walk_forward_classification(data, FEATURE_COLUMNS, "target", folds, horizon=horizon)


def run_final_lockbox(
    df: pd.DataFrame,
    horizon: int = 5,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, list]:
    """Evaluate one untouched chronological final OOS holdout."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be > 0 and < 0.5")

    data = prepare_baseline_data(df, horizon=horizon)
    n_rows = len(data)
    test_size = max(1, int(n_rows * test_fraction))
    train_end = n_rows - test_size - horizon
    if train_end < 2:
        raise ValueError("not enough observations for lockbox train, purge, and test")

    fold = Fold(0, train_end, train_end + horizon, n_rows)
    return walk_forward_classification(data, FEATURE_COLUMNS, "target", [fold], horizon=horizon)


def _final_lockbox_fold(data: pd.DataFrame, horizon: int, test_fraction: float) -> Fold:
    """Create the single final lockbox fold shared by all experiments."""
    n_rows = len(data)
    test_size = max(1, int(n_rows * test_fraction))
    train_end = n_rows - test_size - horizon
    if train_end < 2:
        raise ValueError("not enough observations for lockbox train, purge, and test")
    return Fold(0, train_end, train_end + horizon, n_rows)


def run_final_lockbox_experiments(
    df: pd.DataFrame,
    *,
    horizon: int = 5,
    test_fraction: float = 0.2,
    macro_features: list[str] | None = None,
    geopolitical_features: list[str] | None = None,
) -> list[ExperimentResult]:
    """Run A/B/C feature experiments against one identical untouched lockbox."""
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

    fold = _final_lockbox_fold(data, horizon, test_fraction)
    return run_feature_ablation(
        data,
        [fold],
        macro_features=macro_features,
        geopolitical_features=geopolitical_features,
    )


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
    """Run A/B/C using timestamped events as the geopolitical feature set.

    Event features are generated from the full chronological market index
    before target filtering. ``published_at`` therefore controls exactly when
    information can enter experiment C; later confirmation cannot leak into
    earlier observations.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex for event features")

    event_data = events_to_features(
        df.index,
        events,
        half_life_days=event_half_life_days,
    )
    enriched = df.join(event_data, how="left")
    macro_features = macro_features or []
    if event_features is None:
        event_features = list(event_data.columns)
    missing = [column for column in event_features if column not in enriched.columns]
    if missing:
        raise ValueError(f"Missing event columns: {missing}")

    return run_final_lockbox_experiments(
        enriched,
        horizon=horizon,
        test_fraction=test_fraction,
        macro_features=macro_features,
        geopolitical_features=event_features,
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
        df,
        horizon=horizon,
        test_fraction=test_fraction,
        macro_features=macro_features,
        geopolitical_features=geopolitical_features,
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
        backtest_frame, metrics = backtest_long_only(
            frame,
            threshold=threshold,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            periods_per_year=periods_per_year,
        )
        backtests[experiment.name] = backtest_frame
        rows.append({"experiment": experiment.name, **metrics})

    return pd.DataFrame(rows), backtests
