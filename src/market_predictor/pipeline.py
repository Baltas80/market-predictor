"""Chronological, purged walk-forward baseline pipeline."""

from __future__ import annotations

import pandas as pd

from .backtest import Fold, make_walk_forward_folds, walk_forward_classification
from .experiments import ExperimentResult, run_feature_ablation
from .features import add_market_features, make_target
from .financial import backtest_long_only

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
    data["target"] = make_target(data, horizon=horizon)
    return data.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()


def run_baseline(
    df: pd.DataFrame,
    horizon: int = 5,
    initial_train_fraction: float = 0.6,
    test_fraction: float = 0.1,
) -> tuple[pd.DataFrame, list]:
    """Run expanding-window out-of-sample evaluation with a purge gap.

    The gap equals ``horizon`` so training labels cannot reach into the first
    observations of the test window.
    """
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
    return walk_forward_classification(data, FEATURE_COLUMNS, "target", folds)


def run_final_lockbox(
    df: pd.DataFrame,
    horizon: int = 5,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, list]:
    """Evaluate one untouched chronological final OOS holdout.

    The final ``test_fraction`` of model-ready observations is reserved as a
    lockbox. Training ends before a purge gap of ``horizon`` observations,
    and no expanding-window refits are performed inside the lockbox. This
    function is intended for the final reported estimate after model choices
    and sensitivity settings have been frozen.
    """
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
    return walk_forward_classification(data, FEATURE_COLUMNS, "target", [fold])


def run_final_lockbox_experiments(
    df: pd.DataFrame,
    *,
    horizon: int = 5,
    test_fraction: float = 0.2,
    macro_features: list[str] | None = None,
    geopolitical_features: list[str] | None = None,
) -> list[ExperimentResult]:
    """Run A/B/C feature experiments against one identical untouched lockbox.

    A = technical, B = technical + macro, C = technical + macro +
    geopolitical. All three use the same final chronological test block and
    the same purge gap, making financial comparisons directly comparable.
    """
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

    n_rows = len(data)
    test_size = max(1, int(n_rows * test_fraction))
    train_end = n_rows - test_size - horizon
    if train_end < 2:
        raise ValueError("not enough observations for lockbox train, purge, and test")

    fold = Fold(0, train_end, train_end + horizon, n_rows)
    return run_feature_ablation(
        data,
        [fold],
        macro_features=macro_features,
        geopolitical_features=geopolitical_features,
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
    """Compare A/B/C financial performance on the exact same OOS lockbox.

    Predictions are aligned back to the prepared close series before the
    financial engine is called. The same threshold, costs and annualization
    assumptions are used for every experiment.
    """
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
