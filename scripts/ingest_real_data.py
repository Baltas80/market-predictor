"""Download reproducible real data for Market Predictor.

This legacy source-by-source utility remains useful for manual inspection.
The production historical lockbox uses ``scripts/ingest_historical.py`` plus
its bounded GDELT workflow.

Examples:
  python scripts/ingest_real_data.py --market-symbol '^spx' --start 2000-01-01
  FRED_API_KEY=... python scripts/ingest_real_data.py --fred-series UNRATE CPIAUCSL
  python scripts/ingest_real_data.py --gdelt-start 2015-02-19 --gdelt-end 2015-12-31
  python scripts/ingest_real_data.py --sec

GDELT is downloaded day by day. Raw files and generated inputs belong under
``data/raw`` and are intentionally ignored by Git.
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
    load_sec_litigation_releases_rss,
    load_stooq_daily,
    write_events_csv,
    write_market_csv,
)

GDELT_START = pd.Timestamp("2015-02-19")


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
    parser.add_argument("--sec", action="store_true")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    market = load_stooq_daily(args.market_symbol, args.start, args.end)
    write_market_csv(market, out / f"market_{args.market_symbol.replace('^', 'index_')}.csv")
    print(f"market: {len(market):,} rows")

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
            print(f"fred {series_id}: {len(macro):,} vintage rows")

    if args.gdelt_start:
        if not args.gdelt_end:
            raise SystemExit("--gdelt-end is required with --gdelt-start")
        start = pd.Timestamp(args.gdelt_start)
        end = pd.Timestamp(args.gdelt_end)
        if start < GDELT_START:
            raise SystemExit("GDELT 2.0 Event Database starts on 2015-02-19; earlier dates are not valid source-days")
        if end < start:
            raise SystemExit("--gdelt-end must be on or after --gdelt-start")
        for day in pd.date_range(start, end, freq="D"):
            target = out / f"gdelt_events_{day:%Y%m%d}.csv"
            if target.exists():
                continue
            raw = load_gdelt_day(day)
            events = gdelt_events_to_market_events(raw)
            write_events_csv(events, target)
            print(f"gdelt {day:%Y-%m-%d}: {len(raw):,} raw rows -> {len(events):,} events")
            time.sleep(max(0.0, args.sleep_seconds))

        files = sorted(out.glob("gdelt_events_*.csv"))
        if files:
            combined = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
            for column in ("event_time", "published_at", "available_at"):
                if column in combined.columns:
                    combined[column] = pd.to_datetime(combined[column], utc=True, errors="coerce")
            if "available_at" not in combined.columns or combined["available_at"].isna().all():
                raise SystemExit("GDELT combined data has no valid information-availability timestamps")
            combined = combined.drop_duplicates("event_id").sort_values("available_at")
            write_events_csv(combined, out / "events_gdelt.csv")
            print(f"gdelt combined: {len(combined):,} events")

    if args.sec:
        sec = load_sec_litigation_releases_rss()
        write_events_csv(sec, out / "events_sec_litigation.csv")
        print(f"sec litigation releases: {len(sec):,}")


if __name__ == "__main__":
    main()
