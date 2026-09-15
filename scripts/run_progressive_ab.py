"""Run real-data A/B research without consuming the frozen final OOS lockbox."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from market_predictor.backtest import make_walk_forward_folds
from market_predictor.data_sources import align_fred_point_in_time
from market_predictor.dataset import load_market, materialize_macro
from market_predictor.experiments import run_feature_ablation, summarize_experiments
from market_predictor.features import add_market_features, make_target
from market_predictor.financial import backtest_long_only

FRED_SERIES = ("FEDFUNDS", "DGS10", "CPIAUCSL", "UNRATE", "VIXCLS")
TECHNICAL = [
    "return_1d", "return_5d", "volatility_20d", "price_to_sma20",
    "volume_change", "range_pct",
]
HORIZON = 5
INITIAL_TRAIN_FRACTION = 0.60
TEST_FRACTION = 0.10
TRANSACTION_COST_BPS = 5.0
SLIPPAGE_BPS = 0.0


def sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    staging = Path(os.environ.get("STAGING", "data/historical"))
    output = Path(os.environ.get("OUTPUT", "data/progressive_ab"))
    output.mkdir(parents=True, exist_ok=True)

    market = load_market(staging / "normalized" / "market.csv")
    raw_macro = pd.read_csv(staging / "raw" / "macro_fred.csv")
    fred: dict[str, Path] = {}
    for series_id in FRED_SERIES:
        subset = raw_macro.loc[
            raw_macro["series_id"] == series_id,
            ["observation_date", "value", "vintage_start", "vintage_end"],
        ].rename(
            columns={
                "observation_date": "date",
                "vintage_start": "realtime_start",
                "vintage_end": "realtime_end",
            }
        )
        if subset.empty:
            raise RuntimeError(f"Missing required FRED series: {series_id}")
        path = output / f"fred_{series_id}.csv"
        subset.to_csv(path, index=False)
        fred[series_id] = path

    macro = materialize_macro(market.index, fred, align_fred_point_in_time)
    data = add_market_features(market.join(
        macro.drop(columns=[c for c in macro.columns if c.endswith("_vintage")]),
        how="left",
    ))
    data["target"] = make_target(data, horizon=HORIZON)

    # A and B must use exactly the same admissible observations. Because B
    # contains all five macro series, the common sample starts only where
    # every PIT macro input is actually available; no revised pre-vintage data
    # are backfilled.
    required = TECHNICAL + list(FRED_SERIES) + ["target"]
    data = data.dropna(subset=required).copy()

    initial_train_size = max(1, int(len(data) * INITIAL_TRAIN_FRACTION))
    test_size = max(1, int(len(data) * TEST_FRACTION))
    folds = make_walk_forward_folds(
        len(data),
        initial_train_size=initial_train_size,
        test_size=test_size,
        step=test_size,
        purge=HORIZON,
    )
    if not folds:
        raise RuntimeError("No valid walk-forward folds available")

    results = run_feature_ablation(
        data,
        folds,
        macro_features=list(FRED_SERIES),
        geopolitical_features=[],
        target="target",
    )[:2]
    summary = summarize_experiments(results)
    summary.to_csv(output / "ab_summary.csv", index=False)

    financial_rows = []
    for result in results:
        frame = result.predictions.join(data[["close"]], how="left")
        _, metrics = backtest_long_only(
            frame,
            threshold=0.5,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            slippage_bps=SLIPPAGE_BPS,
            periods_per_year=252,
        )
        financial_rows.append({"experiment": result.name, **metrics})
        frame.to_csv(output / f"predictions_{result.name}.csv")
    financial = pd.DataFrame(financial_rows)
    financial.to_csv(output / "ab_financial.csv", index=False)

    source_files = [
        staging / "normalized" / "market.csv",
        staging / "raw" / "macro_fred.csv",
    ]
    metadata = {
        "experiment_id": "progressive-ab-v1",
        "commit": os.environ.get("GITHUB_SHA", "unknown"),
        "branch": os.environ.get("GITHUB_REF_NAME", "unknown"),
        "dataset_hash": sha256_files(source_files),
        "coverage_start": str(data.index.min()),
        "coverage_end": str(data.index.max()),
        "observations": len(data),
        "features_A": tuple(TECHNICAL),
        "features_B": tuple(TECHNICAL + list(FRED_SERIES)),
        "model": "market_predictor.model.fit_predict",
        "horizon": HORIZON,
        "initial_train_fraction": INITIAL_TRAIN_FRACTION,
        "test_fraction": TEST_FRACTION,
        "folds": len(folds),
        "purge": HORIZON,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "benchmark": "S&P 500 close / buy-and-hold via backtest_long_only",
        "lockbox_used": False,
        "pit_macro": True,
        "common_sample_for_A_B": True,
        "result_files": ["ab_summary.csv", "ab_financial.csv"],
    }
    (output / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(financial.to_string(index=False))
    print(json.dumps(metadata, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
