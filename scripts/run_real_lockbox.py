"""Run the frozen A/B/C lockbox and financial evaluation on staged real data."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_predictor.data_sources import align_fred_point_in_time
from market_predictor.dataset import load_events, load_market, materialize_macro
from market_predictor.final_financial_report import build_final_financial_report, write_financial_report
from market_predictor.pipeline import run_final_lockbox_event_experiments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", default="data/historical")
    parser.add_argument("--output", default="data/results")
    args = parser.parse_args()

    staging = Path(args.staging)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    market = load_market(staging / "normalized" / "market.csv")
    fred = {}
    raw_macro = pd.read_csv(staging / "raw" / "macro_fred.csv")
    for series_id in sorted(raw_macro["series_id"].dropna().unique()):
        path = output / f"fred_{series_id}.csv"
        raw_macro.loc[raw_macro["series_id"] == series_id, ["observation_date", "value", "vintage_start", "vintage_end"]].to_csv(path, index=False)
        fred[series_id] = path

    macro = materialize_macro(market.index, fred, align_fred_point_in_time)
    panel = market.join(macro.drop(columns=[c for c in macro.columns if c.endswith("_vintage")]), how="left")

    event_paths = []
    for name in ("events_gdelt.csv", "events_sec_litigation.csv"):
        path = staging / "normalized" / name
        if path.exists():
            event_paths.append(path)
    events = load_events(event_paths)

    macro_features = [c for c in panel.columns if c in {"FEDFUNDS", "DGS10", "CPIAUCSL", "UNRATE", "VIXCLS"}]
    event_objects = events.to_dict(orient="records")
    from market_predictor.event_schema import MarketEvent
    market_events = [MarketEvent(**row) for row in event_objects]

    results = run_final_lockbox_event_experiments(
        panel,
        market_events,
        horizon=5,
        test_fraction=0.2,
        macro_features=macro_features,
    )

    predictions = {}
    for result in results:
        frame = result.predictions.join(panel[["close"]], how="left")
        predictions[result.name] = frame
        frame.to_csv(output / f"predictions_{result.name}.csv")

    lockbox_index = next(iter(predictions.values())).index
    benchmark = panel.loc[lockbox_index, ["close"]].copy()
    periods = {
        "lockbox_full": (str(lockbox_index.min().date()), str(lockbox_index.max().date())),
    }
    report = build_final_financial_report(
        predictions,
        benchmark=benchmark,
        periods=periods,
        transaction_cost_bps=5.0,
        slippage_bps=0.0,
    )
    write_financial_report(report, output)
    print(f"RESULT_HASH={report.result_hash}")
    print(report.matrix.to_string(index=False))


if __name__ == "__main__":
    main()
