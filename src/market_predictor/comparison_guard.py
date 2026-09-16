"""Guards ensuring experiments are evaluated on the same OOS sample."""
from __future__ import annotations
from typing import Mapping
import pandas as pd
from .reproducibility import canonical_json_hash

def require_common_oos(results: Mapping[str, Mapping[str, object]], required: tuple[str, ...]) -> str:
    if not required: raise ValueError("required experiments cannot be empty")
    missing = [name for name in required if name not in results]
    if missing: raise ValueError(f"missing experiments: {missing}")
    keys = ("oos_start", "oos_end", "purge_gap", "transaction_cost_bps", "slippage_bps", "observations")
    signatures = []
    for name in required:
        row = results[name]
        if any(row.get(k) is None for k in keys): raise ValueError(f"{name} is missing common-protocol metadata")
        signatures.append(tuple(row[k] for k in keys))
    if len(set(signatures)) != 1: raise ValueError("experiments do not share an identical OOS protocol")
    return canonical_json_hash({"protocol": signatures[0], "experiments": required})

def prediction_index_hash(predictions: pd.Series) -> str:
    idx = pd.DatetimeIndex(predictions.index)
    if idx.has_duplicates or not idx.is_monotonic_increasing: raise ValueError("prediction index must be unique and chronological")
    return canonical_json_hash([ts.isoformat() for ts in idx])
