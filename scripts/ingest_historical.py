"""Prepare or execute the declared 2000-2025 historical ingestion plan.

Default mode is dry-run: it validates the coverage plan and writes no market,
macro or event data. ``--execute`` is intentionally explicit because real
source retrieval must be independently reviewed before becoming a lockbox
input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_predictor.historical_coverage import coverage_dict, DATASET_START, DATASET_END


def build_plan(output_dir: Path) -> dict[str, object]:
    return {
        "status": "planned_not_validated",
        "dataset_start": DATASET_START.isoformat(),
        "dataset_end": DATASET_END.isoformat(),
        "coverage": coverage_dict(),
        "raw_directory": str(output_dir / "raw"),
        "normalized_directory": str(output_dir / "normalized"),
        "manifest": str(output_dir / "source_manifest.json"),
        "execution_policy": {
            "default": "dry_run",
            "requires_explicit_execute": True,
            "no_lockbox_admission_from_download_alone": True,
            "historical_gate_required": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Market Predictor historical ingestion planner")
    parser.add_argument("--output-dir", default="data/historical", help="staging directory")
    parser.add_argument("--execute", action="store_true", help="enable source retrieval; downloads are not lockbox validation")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    plan = build_plan(output_dir)
    plan["mode"] = "execute" if args.execute else "dry_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "ingestion_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.execute:
        print("DRY RUN: no external historical data was downloaded.")
        return 0
    raise SystemExit(
        "Explicit retrieval is not yet enabled in this executor. Source adapters must be wired and reviewed before --execute can download data."
    )


if __name__ == "__main__":
    raise SystemExit(main())
