"""Protocol-safe comparison of A/B/C/D and simple baselines."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

EXPERIMENTS = ("A", "B", "C", "D", "buy_and_hold", "random")


def compare_predictions(
    results: Mapping[str, Mapping[str, float]],
    *,
    required_experiments: tuple[str, ...] = EXPERIMENTS,
) -> pd.DataFrame:
    """Build a comparison table while refusing incomplete/mixed protocols.

    Inputs must already be produced under identical OOS dates, purge gap and
    execution costs. This function performs no model selection or tuning.
    """
    if not required_experiments:
        raise ValueError("at least one experiment is required")
    missing = [name for name in required_experiments if name not in results]
    if missing:
        raise ValueError(f"missing comparison results: {missing}")
    rows = []
    protocol_keys = ("oos_start", "oos_end", "purge_gap", "transaction_cost_bps", "slippage_bps")
    reference_protocol = None
    for name in required_experiments:
        row = dict(results[name])
        protocol = tuple(row.get(key) for key in protocol_keys)
        if any(value is None for value in protocol):
            raise ValueError(f"{name} is missing comparison protocol metadata")
        if reference_protocol is None:
            reference_protocol = protocol
        elif protocol != reference_protocol:
            raise ValueError("all experiments must use identical OOS protocol metadata")
        row["experiment"] = name
        rows.append(row)
    return pd.DataFrame(rows).set_index("experiment")
