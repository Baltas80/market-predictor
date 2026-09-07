"""Download and persist one bounded GDELT historical chunk.

Each chunk is independently reproducible so a failed/cancelled CI job never
requires re-downloading earlier completed chunks.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from market_predictor.historical_adapters import fetch_gdelt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = fetch_gdelt(args.start, args.end)
    frame.to_csv(output, index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z")
    print(f"GDELT chunk saved: {args.start} -> {args.end}; rows={len(frame)}; path={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
