"""Download reproducible real data for Market Predictor.

Examples:
  python scripts/ingest_real_data.py --market-symbol '^spx' --start 2000-01-01
  FRED_API_KEY=... python scripts/ingest_real_data.py --fred-series UNRATE CPIAUCSL
  python scripts/ingest_real_data.py --gdelt-start 2020-01-01 --gdelt-end 2020-12-31

GDELT can be very large. It is intentionally opt-in and downloaded day by
 day so a failed run can be resumed without corrupting a monolithic archive.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import pandas as pd

from market_predictor.data_sources import (
    gdelt_events_to_market_events,
    load_fred_observations,
    load_gdelt_day,
    load_stooq_daily,
    write_events_csv,
    write_market_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-symbol", default="^spx")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--fred-series", nargs="*", default=[])
    parser.add_argument("--fred-realtime-start", default=None)
    parser.add_argument("--fred-realtime-end", default=None)
    parser.add_argument("--gdelt-start", default=None)
    parser.add_argument("--gdelt-end", default=None)
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    market = load_stooq_daily(args.market_symbol, args.start, args.end)
    write_market_csv(market, out / f"market_{args.market_symbol.replace('^', 'index_')}.csv")

    if args.fred_series:
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise SystemExit("FRED_API_KEY is required when --fred-series is used")
        for series_id in args.fred_series:
            macro = load_fred_observations(
                series_id,
                api_key,
                realtime_start=args.fred_realtime_start,
                realtime_end=args.fred_realtime_end,
            )
            macro.to_csv(out / f"fred_{series_id}.csv", index=False)

    if args.gdelt_start:
        if not args.gdelt_end:
            raise SystemExit("--gdelt-end is required with --gdelt-start")
        start = pd.Timestamp(args.gdelt_start)
        end = pd.Timestamp(args.gdelt_end)
        if end < start:
            raise SystemExit("--gdelt-end must be on or after --gdelt-start")
        chunks: list[pd.DataFrame] = []
        for day in pd.date_range(start, end, freq="D"):
            target = out / f"gdelt_events_{day:%Y%m%d}.csv"
            if target.exists():
                continue
            try:
                raw = load_gdelt_day(day)
                events = gdelt_events_to_market_events(raw)
                write_events_csv(events, target)
                chunks.append(events)
            except Exception as exc:
                print(f"GDELT {day:%Y-%m-%d}: {exc}")
            time.sleep(max(0.0, args.sleep_seconds))

        # Build one deterministic deduplicated event file from all downloaded days.
        files = sorted(out.glob("gdelt_events_*.csv"))
        if files:
            combined = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
            combined["published_at"] = pd.to_datetime(combined["published_at"], utc=True)
            combined = combined.drop_duplicates("event_id").sort_values("published_at")
            combined["published_at"] = combined["published_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            write_events_csv(combined, out / "events_gdelt.csv")


if __name__ == "__main__":
    main()
