"""Run real A/B on a bounded, common market+PIT-FRED window.

This deliberately avoids the full historical staging pipeline so A/B research can
start before GDELT 1.0 completion. A uses technical market features only; B adds
only the PIT-vintage FRED series. Both share identical rows, chronological folds,
purge, costs, execution rules and financial backtest.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from market_predictor.backtest import make_walk_forward_folds
from market_predictor.data_sources import align_fred_point_in_time
from market_predictor.experiments import run_feature_ablation, summarize_experiments
from market_predictor.financial import backtest_long_only
from market_predictor.features import add_market_features, make_target
from market_predictor.historical_adapters import fetch_fred, fetch_market, FRED_SERIES

TECHNICAL = [
    "return_1d", "return_5d", "volatility_20d", "price_to_sma20",
    "volume_change", "range_pct",
]
HORIZON = 5
INITIAL_TRAIN_FRACTION = 0.60
TEST_FRACTION = 0.10
TRANSACTION_COST_BPS = 5.0
SLIPPAGE_BPS = 0.0
PERIODS_PER_YEAR = 252


def sha256_frame(frame: pd.DataFrame) -> str:
    raw = frame.sort_index().to_csv(index=True, lineterminator="\n").encode()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    start = os.environ.get("AB_START", "2015-01-01")
    end = os.environ.get("AB_END", "2025-12-31")
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is required for real PIT macro staging")

    out = Path(os.environ.get("OUTPUT", "data/real_ab_window"))
    out.mkdir(parents=True, exist_ok=True)

    market = fetch_market(start, end)
    if market.empty:
        raise RuntimeError("Market adapter returned no observations")
    market.to_csv(out / "market.csv", lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z")

    macro = fetch_fred(api_key, start, end)
    if macro.empty:
        raise RuntimeError("FRED returned no PIT observations")
    macro.to_csv(out / "macro_fred.csv", index=False, lineterminator="\n")

    fred_paths: dict[str, Path] = {}
    for series_id in FRED_SERIES:
        subset = macro.loc[
            macro["series_id"] == series_id,
            ["observation_date", "value", "vintage_start", "vintage_end"],
        ].rename(
            columns={
                "observation_date": "date",
                "vintage_start": "realtime_start",
                "vintage_end": "realtime_end",
            }
        )
        if subset.empty:
            raise RuntimeError(f"Missing PIT FRED series: {series_id}")
        path = out / f"fred_{series_id}.csv"
        subset.to_csv(path, index=False, lineterminator="\n")
        fred_paths[series_id] = path

    macro_panel = pd.DataFrame(index=market.index)
    for series_id, path in fred_paths.items():
        aligned = align_fred_point_in_time(pd.read_csv(path), market.index)
        macro_panel[series_id] = aligned["value"].to_numpy()

    data = add_market_features(market.join(macro_panel, how="left"))
    data["target"] = make_target(data, horizon=HORIZON)
    required = TECHNICAL + list(FRED_SERIES) + ["target"]
    data = data.dropna(subset=required).copy()
    if len(data) < 100:
        raise RuntimeError(f"Too few common PIT observations for A/B: {len(data)}")

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
        raise RuntimeError("No valid purged walk-forward folds available")

    results = run_feature_ablation(
        data,
        folds,
        macro_features=list(FRED_SERIES),
        geopolitical_features=[],
        target="target",
    )[:2]
    summary = summarize_experiments(results)
    summary.to_csv(out / "ab_summary.csv", index=False, lineterminator="\n")

    financial_rows = []
    for result in results:
        predictions = result.predictions.join(data[["close"]], how="left")
        equity, metrics = backtest_long_only(
            predictions,
            threshold=0.5,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            slippage_bps=SLIPPAGE_BPS,
            periods_per_year=PERIODS_PER_YEAR,
        )
        equity.to_csv(out / f"financial_{result.name}.csv", lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z")
        financial_rows.append({"experiment": result.name, **metrics})
    financial = pd.DataFrame(financial_rows)
    financial.to_csv(out / "ab_financial.csv", index=False, lineterminator="\n")

    metadata = {
        "experiment_id": "real-ab-window-2015-2025-v1",
        "commit": os.environ.get("GITHUB_SHA", "unknown"),
        "branch": os.environ.get("GITHUB_REF_NAME", "unknown"),
        "requested_start": start,
        "requested_end": end,
        "common_sample_start": str(data.index.min()),
        "common_sample_end": str(data.index.max()),
        "observations": int(len(data)),
        "features_A": TECHNICAL,
        "features_B": TECHNICAL + list(FRED_SERIES),
        "model": "market_predictor.model.fit_predict via walk_forward_classification",
        "horizon": HORIZON,
        "initial_train_fraction": INITIAL_TRAIN_FRACTION,
        "test_fraction": TEST_FRACTION,
        "folds": len(folds),
        "purge": HORIZON,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "benchmark": "SPX buy-and-hold over each common OOS prediction path",
        "pit_macro": True,
        "lockbox_used": False,
        "market_hash": sha256_frame(market.loc[data.index.min():data.index.max()]),
        "macro_hash": hashlib.sha256(macro.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
        "result_files": ["ab_summary.csv", "ab_financial.csv"],
    }
    (out / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("=== A/B SUMMARY ===")
    print(summary.to_string(index=False))
    print("=== A/B FINANCIAL ===")
    print(financial.to_string(index=False))
    print("=== METADATA ===")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
