"""Benchmark sequential vs bounded-parallel GDELT 1.0 ingestion.

Experimental only: this never writes to the production lockbox or changes
the production C path. Both modes use the same canonical GDELT 1.0 loader,
same retry policy, same date range and deterministic per-day fingerprint.
The only experimental difference is execution concurrency.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path
import time

import pandas as pd

from market_predictor.gdelt1 import GDELT_SOURCE_ID, load_gdelt_day

RETRIES = 4
RETRY_BASE_SECONDS = 2


@dataclass(frozen=True)
class DayResult:
    date: str
    status: str
    rows: int
    sha256: str | None
    seconds: float
    error: str | None = None


def _stable_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values("global_event_id", kind="mergesort").copy()
    ordered = ordered.sort_index(axis=1)
    payload = ordered.to_csv(index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z").encode("utf-8")
    return sha256(payload).hexdigest()


def _load_with_retries(day: str) -> DayResult:
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            frame = load_gdelt_day(day)
            if frame.empty:
                raise ValueError("empty GDELT 1.0 export")
            if set(frame["source_id"].dropna().astype(str)) != {GDELT_SOURCE_ID}:
                raise ValueError("non-canonical GDELT source_id returned")
            elapsed = time.perf_counter() - started
            return DayResult(day, "ok", len(frame), _stable_hash(frame), elapsed)
        except Exception as exc:  # benchmark must report failure instead of hiding it
            last_error = exc
            if attempt < RETRIES:
                time.sleep(RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
    return DayResult(
        day,
        "failed",
        0,
        None,
        time.perf_counter() - started,
        f"{type(last_error).__name__}: {last_error}",
    )


def _days(start: str, end: str) -> list[str]:
    first = pd.Timestamp(start).date()
    last = pd.Timestamp(end).date()
    if last < first:
        raise ValueError("end must be on or after start")
    out: list[str] = []
    current = first
    while current <= last:
        out.append(current.isoformat())
        current += timedelta(days=1)
    return out


def _run_sequential(days: list[str]) -> list[DayResult]:
    return [_load_with_retries(day) for day in days]


def _run_parallel(days: list[str], workers: int) -> list[DayResult]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_load_with_retries, day): day for day in days}
        results = []
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item.date)


def _compare(slow: list[DayResult], fast: list[DayResult]) -> dict:
    slow_map = {item.date: item for item in slow}
    fast_map = {item.date: item for item in fast}
    dates = sorted(set(slow_map) | set(fast_map))
    rows = []
    mismatches = []
    for day in dates:
        s = slow_map.get(day)
        f = fast_map.get(day)
        same_status = s is not None and f is not None and s.status == f.status
        same_rows = s is not None and f is not None and s.rows == f.rows
        same_hash = s is not None and f is not None and s.sha256 == f.sha256
        identical = same_status and same_rows and same_hash
        rows.append({
            "date": day,
            "slow_status": s.status if s else "missing",
            "fast_status": f.status if f else "missing",
            "slow_rows": s.rows if s else None,
            "fast_rows": f.rows if f else None,
            "slow_sha256": s.sha256 if s else None,
            "fast_sha256": f.sha256 if f else None,
            "identical": identical,
        })
        if not identical:
            mismatches.append(day)
    slow_seconds = sum(item.seconds for item in slow)
    fast_seconds = sum(item.seconds for item in fast)
    slow_ok = sum(item.status == "ok" for item in slow)
    fast_ok = sum(item.status == "ok" for item in fast)
    slow_fail = len(slow) - slow_ok
    fast_fail = len(fast) - fast_ok
    return {
        "all_days_identical": not mismatches,
        "mismatched_days": mismatches,
        "slow_ok": slow_ok,
        "slow_failed": slow_fail,
        "fast_ok": fast_ok,
        "fast_failed": fast_fail,
        "slow_sum_day_seconds": slow_seconds,
        "fast_sum_day_seconds": fast_seconds,
        "speedup_lower_bound": slow_seconds / fast_seconds if fast_seconds > 0 else None,
        "per_day": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="data/experimental/gdelt1_fast_compare")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    days = _days(args.start, args.end)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    slow_started = time.perf_counter()
    slow = _run_sequential(days)
    slow_wall = time.perf_counter() - slow_started

    fast_started = time.perf_counter()
    fast = _run_parallel(days, args.workers)
    fast_wall = time.perf_counter() - fast_started

    comparison = _compare(slow, fast)
    summary = {
        "experiment": "gdelt1-fast-vs-slow",
        "source_id": GDELT_SOURCE_ID,
        "start": args.start,
        "end": args.end,
        "days": len(days),
        "workers": args.workers,
        "slow_wall_seconds": slow_wall,
        "fast_wall_seconds": fast_wall,
        "wall_speedup": slow_wall / fast_wall if fast_wall > 0 else None,
        "comparison": comparison,
    }

    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame([asdict(item) for item in slow]).to_csv(output / "slow_results.csv", index=False)
    pd.DataFrame([asdict(item) for item in fast]).to_csv(output / "fast_results.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if comparison["all_days_identical"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
