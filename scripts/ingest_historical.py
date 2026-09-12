"""Prepare or execute the declared 2000-2025 historical ingestion plan.

Default mode is dry-run. ``--execute`` runs the real Stooq/FRED/GDELT/SEC
adapters and writes a reproducible staging area. Downloaded data is never
admitted to the lockbox without the Historical Gate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from market_predictor.historical_adapters import FRED_SERIES
from market_predictor.historical_coverage import coverage_dict, DATASET_START, DATASET_END, GDELT_SOURCE
from market_predictor.historical_staging import STAGING_VERSION, stage_historical


STAGES = (
    "download",
    "raw",
    "hash",
    "normalize",
    "pit",
    "historical_gate",
    "dataset",
)


def build_plan(output_dir: Path) -> dict[str, object]:
    return {
        "status": "planned_not_validated",
        "dataset_start": DATASET_START.isoformat(),
        "dataset_end": DATASET_END.isoformat(),
        "coverage": coverage_dict(),
        "raw_directory": str(output_dir / "raw"),
        "normalized_directory": str(output_dir / "normalized"),
        "manifest": str(output_dir / "source_manifest.json"),
        "staging_version": STAGING_VERSION,
        "adapters": {
            "market": "Stooq SPX daily",
            "macro": f"FRED {', '.join(FRED_SERIES)} realtime vintages",
            "events": f"GDELT daily exports from {GDELT_SOURCE.start.isoformat()}; SEC litigation RSS snapshot",
        },
        "stages": [{"name": name, "status": "blocked"} for name in STAGES],
        "execution_policy": {
            "default": "dry_run",
            "requires_explicit_execute": True,
            "no_lockbox_admission_from_download_alone": True,
            "historical_gate_required": True,
            "fred_api_key_required": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Market Predictor historical ingestion planner")
    parser.add_argument("--output-dir", default="data/historical", help="staging directory")
    parser.add_argument("--execute", action="store_true", help="run real source adapters")
    parser.add_argument("--skip-gdelt", action="store_true", help="skip the large GDELT historical range")
    parser.add_argument("--skip-sec", action="store_true", help="skip the SEC RSS snapshot")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    plan = build_plan(output_dir)
    plan["mode"] = "execute" if args.execute else "dry_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "ingestion_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        print("DRY RUN: no external historical data was downloaded.")
        return 0

    result = stage_historical(
        output_dir,
        fred_api_key=os.environ.get("FRED_API_KEY"),
        include_gdelt=not args.skip_gdelt,
        include_sec=not args.skip_sec,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
