"""Recover missing GDELT 1.0 daily source-days."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from ingest_gdelt_daily_chunk import GDELT_SOURCE_ID, load_gdelt_daily_day

COLUMNS = ["date", "source_id", "status", "error_type", "error"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--remaining-output", required=True)
    args = parser.parse_args()
    chunks = Path(args.chunks)
    manifests = []
    for path in sorted(chunks.glob("gdelt_*_missing.csv")):
        if path.stat().st_size:
            frame = pd.read_csv(path)
            if not frame.empty:
                manifests.append(frame)
    missing = pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame(columns=COLUMNS)
    rows, remaining = [], []
    for day in sorted(pd.to_datetime(missing["date"], errors="coerce").dropna().dt.date.unique()):
        try:
            rows.append(load_gdelt_daily_day(day))
        except Exception as exc:
            remaining.append({"date": day.isoformat(), "source_id": GDELT_SOURCE_ID, "status": "missing", "error_type": type(exc).__name__, "error": str(exc)})
    recovered = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    recovered.to_csv(args.output, index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z")
    pd.DataFrame(remaining, columns=COLUMNS).to_csv(args.remaining_output, index=False, lineterminator="\n")
    print(f"GDELT 1.0 recovery: recovered_rows={len(recovered)} remaining_days={len(remaining)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
