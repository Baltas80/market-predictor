"""Assemble immutable GDELT 2.0 BigQuery availability chunks.

The chunks are disjoint by DATEADDED availability windows, so finalization
concatenates them after validating names, schema, and availability bounds.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
from pathlib import Path

import pandas as pd

SOURCE_ID = "GDELT_2_Event_Database_BigQuery"
COLUMNS = [
    "event_id", "event_time", "published_at", "available_at", "source_id",
    "category", "severity", "country", "entity", "sector", "duration_days",
    "media_intensity", "surprise",
]


def _expected_chunks(start: dt.date, end: dt.date, days: int = 28) -> list[tuple[int, dt.date, dt.date]]:
    rows = []
    cur = start
    number = 1
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=days - 1), end)
        rows.append((number, cur, chunk_end))
        cur = chunk_end + dt.timedelta(days=1)
        number += 1
    return rows


def _validate_chunk(path: Path, start: dt.date, end: dt.date) -> int:
    frame = pd.read_csv(path, compression="infer")
    if list(frame.columns) != COLUMNS:
        raise RuntimeError(f"Unexpected GDELT2 columns in {path.name}")
    if frame.empty:
        raise RuntimeError(f"Empty GDELT2 chunk: {path.name}")
    available = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
    if available.isna().any():
        raise RuntimeError(f"Invalid available_at values in {path.name}")
    if (available.dt.date < start).any() or (available.dt.date > end).any():
        raise RuntimeError(f"Availability escaped declared chunk bounds in {path.name}")
    if (frame["source_id"] != SOURCE_ID).any():
        raise RuntimeError(f"Unexpected source_id in {path.name}")
    return len(frame)


def _append_gzip_csv(src: Path, dst: Path, include_header: bool) -> int:
    rows = 0
    with gzip.open(src, "rt", encoding="utf-8", newline="") as source, gzip.open(
        dst, "at", encoding="utf-8", newline=""
    ) as target:
        reader = csv.reader(source)
        writer = csv.writer(target, lineterminator="\n")
        header = next(reader)
        if include_header:
            writer.writerow(header)
        for row in reader:
            writer.writerow(row)
            rows += 1
    return rows


def finalize(chunks: Path, output: Path, start: dt.date, end: dt.date) -> None:
    expected = _expected_chunks(start, end)
    selected: list[Path] = []
    for number, chunk_start, chunk_end in expected:
        candidates = sorted(chunks.glob(f"gdelt2_{number}_{chunk_start}_{chunk_end}.csv*"))
        if not candidates:
            raise RuntimeError(
                f"Missing completed GDELT2 chunk {number}: {chunk_start} -> {chunk_end}"
            )
        path = candidates[-1]
        _validate_chunk(path, chunk_start, chunk_end)
        selected.append(path)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    total = 0
    for index, path in enumerate(selected):
        total += _append_gzip_csv(path, output, include_header=(index == 0))

    missing = output.parent / "events_gdelt_missing.csv"
    pd.DataFrame(columns=["date", "source_id", "status", "error_type", "error"]).to_csv(
        missing, index=False, lineterminator="\n"
    )
    print(
        f"GDELT2 finalization passed: chunks={len(selected)} rows={total} "
        f"availability={start}..{end} output={output}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output", default="data/historical/normalized/events_gdelt.csv.gz")
    parser.add_argument("--start", default="2015-02-19")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()
    finalize(
        Path(args.chunks),
        Path(args.output),
        pd.Timestamp(args.start).date(),
        pd.Timestamp(args.end).date(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
