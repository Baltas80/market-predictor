"""Protocol-safe comparison of A/B/C/D and simple baselines."""
from __future__ import annotations

import hashlib

import pandas as pd


EXPERIMENTS = ("A", "B", "C", "D", "buy_and_hold", "random")


def prediction_index_hash(index: pd.DatetimeIndex) -> str:
    """Return a deterministic hash of the exact prediction timestamps."""
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("prediction index must be a DatetimeIndex")
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("prediction index must be valid, unique and chronological")
    return hashlib.sha256("|".join(str(x.value) for x in index).encode()).hexdigest()


# Backwards-compatible private alias used by earlier callers.
def _index_hash(value) -> str:
    return prediction_index_hash(value)


def compare_predictions(results, *, required_experiments=EXPERIMENTS) -> pd.DataFrame:
    """Compare results only when the full evaluation protocol is identical."""
    missing = [name for name in required_experiments if name not in results]
    if missing:
        raise ValueError(f"missing comparison results: {missing}")

    keys = (
        "oos_start",
        "oos_end",
        "purge_gap",
        "transaction_cost_bps",
        "slippage_bps",
        "observations",
        "prediction_index_hash",
    )
    reference = None
    rows = []
    for name in required_experiments:
        row = dict(results[name])
        for key in keys:
            if row.get(key) is None:
                raise ValueError(f"comparison result {name} missing protocol field: {key}")
        if reference is None:
            reference = tuple(row[k] for k in keys)
        elif tuple(row[k] for k in keys) != reference:
            raise ValueError(f"comparison protocol mismatch for {name}")
        rows.append({"experiment": name, **row})
    return pd.DataFrame(rows).set_index("experiment")
