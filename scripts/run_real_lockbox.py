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
from market_predictor.gdelt1 import GDELT_SOURCE_ID
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


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


def _assert_required_gdelt_provenance(staging: Path) -> None:
    """Fail closed unless final staging proves canonical GDELT 1.0 provenance."""
    missing_path = staging / "raw" / "events_gdelt_missing.csv"
    if not missing_path.exists():
        raise RuntimeError(
            "Final lockbox refused: required GDELT gap manifest is missing; "
            "completeness cannot be demonstrated."
        )
    missing = pd.read_csv(missing_path)
    if not missing.empty:
        if "source_id" not in missing.columns:
            raise RuntimeError(
                "Final lockbox refused: GDELT gap manifest has no source_id; "
                "completeness cannot be attributed to the canonical source."
            )
        required = missing.loc[missing["source_id"] == GDELT_SOURCE_ID]
        noncanonical = missing.loc[missing["source_id"] != GDELT_SOURCE_ID]
        if not noncanonical.empty:
            raise RuntimeError(
                "Final lockbox refused: GDELT gap manifest contains non-canonical "
                f"source identifiers: {sorted(set(noncanonical['source_id'].astype(str)))}"
            )
        if not required.empty:
            raise RuntimeError(
                f"Final lockbox refused: {len(required)} required GDELT source-days remain missing. "
                "Resolve the historical ingestion gaps before generating A/B/C or the financial report."
            )

    manifest_path = staging / "source_manifest.json"
    result_path = staging / "staging_result.json"
    if not manifest_path.exists() or not result_path.exists():
        raise RuntimeError(
            "Final lockbox refused: source_manifest.json and staging_result.json "
            "are required provenance artifacts."
        )

    try:
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        staging_result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Final lockbox refused: provenance artifacts are unreadable: {type(exc).__name__}"
        ) from exc

    gdelt_manifests = [
        item for item in source_manifest
        if item.get("source_id") == GDELT_SOURCE_ID
    ]
    noncanonical_manifest_ids = sorted(
        {
            str(item.get("source_id"))
            for item in source_manifest
            if str(item.get("source_id", "")).startswith("GDELT_")
            and item.get("source_id") != GDELT_SOURCE_ID
        }
    )
    if noncanonical_manifest_ids:
        raise RuntimeError(
            "Final lockbox refused: source manifest contains non-canonical GDELT "
            f"identifiers: {noncanonical_manifest_ids}"
        )
    if len(gdelt_manifests) != 1:
        raise RuntimeError(
            "Final lockbox refused: expected exactly one canonical GDELT 1.0 "
            f"source manifest, found {len(gdelt_manifests)}."
        )

    policy = str(gdelt_manifests[0].get("availability_policy", ""))
    if "next-day 06:00 America/New_York" not in policy:
        raise RuntimeError(
            "Final lockbox refused: canonical GDELT 1.0 PIT availability policy "
            "is not declared as the approved next-day 06:00 America/New_York boundary."
        )

    if staging_result.get("status") not in {"admissible", "admissible_with_source_limits"}:
        raise RuntimeError(
            "Final lockbox refused: staging result is not admissible: "
            f"{staging_result.get('status')!r}"
        )
    if staging_result.get("gdelt_missing_day_count") != 0:
        raise RuntimeError(
            "Final lockbox refused: staging result reports non-zero GDELT missing-day count."
        )
    if staging_result.get("gdelt_chunk_count") != staging_result.get("gdelt_expected_chunk_count"):
        raise RuntimeError(
            "Final lockbox refused: staged GDELT chunk count does not match the expected matrix."
        )


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

    _assert_required_gdelt_provenance(staging)
    dataset_hash = _staging_fingerprint(staging)
    source_manifest_hash = _file_sha256(staging / "source_manifest.json")
    staging_result_hash = _file_sha256(staging / "staging_result.json")

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
    panel = market.join(
        macro.drop(columns=[c for c in macro.columns if c.endswith("_vintage")]),
        how="left",
    )

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
        result_hashes=tuple(
            prediction_hashes
            + [
                ("financial_report", report.result_hash),
                ("source_manifest", source_manifest_hash),
                ("staging_result", staging_result_hash),
            ]
        ),
    )
    manifest_path = output / "lockbox_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.as_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"RESULT_HASH={report.result_hash}")
    print(f"LOCKBOX_MANIFEST_HASH={manifest.fingerprint()}")
    print(report.matrix.to_string(index=False))
    print(report.stability.to_string(index=False))


if __name__ == "__main__":
    main()
