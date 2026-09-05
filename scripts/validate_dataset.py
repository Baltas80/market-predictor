"""Run deterministic dataset integrity checks and write a JSON report.

Usage examples:
  python scripts/validate_dataset.py --market data/raw/market.csv
  python scripts/validate_dataset.py --market data/raw/market.csv --events data/raw/events.csv --report data/quality_report.json

The command fails closed: if any supplied dataset fails validation, the process
exits non-zero. It never repairs or silently drops invalid records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_predictor.data_quality import quality_report
from market_predictor.dataset import load_events, load_market, load_fred_vintage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", type=Path)
    parser.add_argument("--fred", action="append", default=[], help="FRED/ALFRED CSV; repeat for multiple series")
    parser.add_argument("--events", type=Path)
    parser.add_argument("--report", type=Path, default=Path("data/quality_report.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not any((args.market, args.fred, args.events)):
        raise SystemExit("At least one input (--market, --fred or --events) is required")

    report_parts = {}
    if args.market:
        report_parts["market"] = quality_report(market=load_market(args.market))
    if args.fred:
        for path in args.fred:
            report_parts[f"fred:{path}"] = quality_report(macro=load_fred_vintage(path))
    if args.events:
        report_parts["events"] = quality_report(events=load_events([args.events]))

    passed = all(part["passed"] for part in report_parts.values())
    report = {"passed": passed, "datasets": report_parts}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
