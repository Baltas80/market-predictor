"""Materialize a deterministic, point-in-time research dataset.

This module joins already-downloaded source files. It never downloads data and
never mutates the final OOS lockbox. Raw source data remains outside Git.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pandas as pd


DEFAULT_EVENT_COLUMNS = [
    "event_id", "category", "published_at", "severity", "country", "entity",
    "sector", "duration_days", "media_intensity", "surprise",
]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_market(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Market file missing columns: {sorted(missing)}")
    frame = frame.set_index("date").sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Market data contains duplicate dates")
    return frame


def load_fred_vintage(path: str | Path) -> pd.DataFrame:
    """Load FRED vintages while allowing realtime_end to be absent.

    realtime_start is the minimum metadata required for point-in-time alignment;
    realtime_end is optional because exported/test vintage tables may omit it.
    """
    frame = pd.read_csv(path)
    required = {"date", "value", "realtime_start"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"FRED file missing columns: {sorted(missing)}")
    for column in ("date", "realtime_start", "realtime_end"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["date", "value", "realtime_start"])


def materialize_macro(
    market_index: pd.DatetimeIndex,
    fred_files: dict[str, str | Path],
    aligner,
) -> pd.DataFrame:
    """Create a market-date macro panel using only historical vintages."""
    output = pd.DataFrame(index=pd.DatetimeIndex(market_index).normalize())
    for series_id, path in fred_files.items():
        observations = load_fred_vintage(path)
        aligned = aligner(observations, output.index)
        output[series_id] = aligned["value"].to_numpy()
        output[f"{series_id}_vintage"] = aligned["vintage"].to_numpy()
    return output


def load_events(paths: list[str | Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = set(DEFAULT_EVENT_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Event file {path} missing columns: {sorted(missing)}")
        frames.append(frame[DEFAULT_EVENT_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=DEFAULT_EVENT_COLUMNS)
    events = pd.concat(frames, ignore_index=True)
    events["published_at"] = pd.to_datetime(events["published_at"], utc=True, errors="coerce")
    events["severity"] = pd.to_numeric(events["severity"], errors="coerce")
    events = events.dropna(subset=["event_id", "published_at", "severity"])
    events = events.drop_duplicates("event_id").sort_values("published_at")
    return events.reset_index(drop=True)


def build_manifest(paths: list[str | Path], *, dataset_name: str, start: str | None, end: str | None) -> dict:
    files = []
    for path in sorted(map(Path, paths), key=lambda item: str(item)):
        files.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {
        "dataset": dataset_name,
        "start": start,
        "end": end,
        "files": files,
        "point_in_time": True,
    }


def write_manifest(manifest: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
