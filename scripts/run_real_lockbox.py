"""Run the frozen A/B/C lockbox and financial evaluation on staged real data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from market_predictor.data_sources import align_fred_point_in_time
from market_predictor.dataset import load_market, materialize_macro
from market_predictor.event_io import load_events_csv
from market_predictor.final_financial_report import build_final_financial_report, write_financial_report
from market_predictor.lockbox_manifest import LockboxManifest
from market_predictor.pipeline import PROTOCOL_VERSION, run_final_lockbox_event_experiments
from market_predictor.reproducibility import canonical_json_hash


FRED_SERIES = ("FEDFUNDS", "DGS10", "CPIAUCSL", "UNRATE", "VIXCLS")


def _event_sort_key(event) -> pd.Timestamp:
    """Order events by information availability, not unknown publication time."""
    available = pd.Timestamp(event.available_at)
    if pd.notna(available):
        return available
    published = pd.Timestamp(event.published_at)
    if pd.notna(published):
        return published
    return pd.Timestamp(event.event_time)


def _staging_fingerprint(staging: Path) -> str:
    """Hash every normalized input file used by the final lockbox."""
    paths = [
        staging / "normalized" / "market.csv",
        staging / "raw" / "macro_fred.csv",
        staging / "normalized" / "events_gdelt.csv",
        staging / "normalized" / "events_sec_litigation.csv",
    ]
    digest = hashlib.sha256()
    found = False
    for path in sorted(paths, key=lambda item: str(item)):
        if not path.exists() or path.stat().st_size == 0:
            continue
        found = True
        digest.update(str(path.relative_to(staging)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    if not found:
        raise RuntimeError("No normalized staging inputs available for dataset fingerprint")
    return digest.hexdigest()


def _prediction_hash(frame: pd.DataFrame) -> str:
    """Hash the complete OOS prediction frame deterministically."""
    records = frame.reset_index().astype(object).where(pd.notna(frame), None).to_dict(orient="records")
    return canonical_json_hash(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", default="data/historical")
    parser.add_argument("--output", default="data/results")
    args = parser.parse_args()

    staging = Path(args.staging)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    dataset_hash = _staging_fingerprint(staging)

    market = load_market(staging / "normalized" / "market.csv")
    raw_macro = pd.read_csv(staging / "raw" / "macro_fred.csv")
    fred: dict[str, Path] = {}
    for series_id in FRED_SERIES:
        subset = raw_macro.loc[
            raw_macro["series_id"] == series_id,
            ["observation_date", "value", "vintage_start", "vintage_end"],
        ]
        if subset.empty:
            raise RuntimeError(f"Missing required FRED series in staged data: {series_id}")
        path = output / f"fred_{series_id}.csv"
        subset.to_csv(path, index=False)
        fred[series_id] = path

    macro = materialize_macro(market.index, fred, align_fred_point_in_time)
    panel = market.join(macro.drop(columns=[c for c in macro.columns if c.endswith("_vintage")]), how="left")

    events = []
    for name in ("events_gdelt.csv", "events_sec_litigation.csv"):
        path = staging / "normalized" / name
        if path.exists() and path.stat().st_size > 0:
            events.extend(load_events_csv(path))
    events.sort(key=_event_sort_key)
    if not events:
        raise RuntimeError("No staged historical events available for experiment C")

    results = run_final_lockbox_event_experiments(
        panel,
        events,
        horizon=5,
        test_fraction=0.2,
        macro_features=list(FRED_SERIES),
    )

    predictions: dict[str, pd.DataFrame] = {}
    prediction_hashes: list[tuple[str, str]] = []
    for result in results:
        frame = result.predictions.join(panel[["close"]], how="left")
        predictions[result.name] = frame
        prediction_hashes.append((result.name, _prediction_hash(frame)))
        frame.to_csv(output / f"predictions_{result.name}.csv")

    lockbox_index = next(iter(predictions.values())).index
    benchmark = panel.loc[lockbox_index, ["close"]].copy()
    periods = {
        "lockbox_early": ("2021-01-01", "2022-12-31"),
        "lockbox_middle": ("2023-01-01", "2024-12-31"),
        "lockbox_late": ("2025-01-01", "2025-12-31"),
    }
    report = build_final_financial_report(
        predictions,
        benchmark=benchmark,
        periods=periods,
        transaction_cost_bps=5.0,
        slippage_bps=0.0,
    )
    write_financial_report(report, output)

    manifest = LockboxManifest(
        oos_start=pd.Timestamp(lockbox_index[0]).date(),
        oos_end=pd.Timestamp(lockbox_index[-1]).date(),
        purge_gap=5,
        dataset_hash=dataset_hash,
        code_version=os.environ.get("GITHUB_SHA", "manual-local-run"),
        protocol_version=PROTOCOL_VERSION,
        transaction_cost_bps=5.0,
        slippage_bps=0.0,
        result_hashes=tuple(prediction_hashes + [("financial_report", report.result_hash)]),
    )
    manifest_path = output / "lockbox_manifest.json"
    manifest_path.write_text(json.dumps(manifest.as_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"RESULT_HASH={report.result_hash}")
    print(f"LOCKBOX_MANIFEST_HASH={manifest.fingerprint()}")
    print(report.matrix.to_string(index=False))
    print(report.stability.to_string(index=False))


if __name__ == "__main__":
    main()
