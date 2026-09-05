"""Protocol-safe comparison of A/B/C/D and simple baselines."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

EXPERIMENTS = ("A", "B", "C", "D", "buy_and_hold", "random")


def compare_predictions(
    results: Mapping[str, Mapping[str, float]],
    *,
    required_experiments: tuple[str, ...] = ("A", "B", "C", "D", "buy_and_hold", "random"),
) -> pd.DataFrame:
    """Build a comparison table while refusing incomplete/mixed protocols.

    Inputs must already be produced under the same OOS dates, purge gap and costs;
    this function deliberately performs no model selection or tuning.
    """
    missing = [name for name in required_experiments if name not in results]
    if missing:
        raise ValueError(f"missing comparison results: {missing}")
    rows = []
    for name in required_experiments:
        row = {"experiment": name, **dict(results[name])}
        rows.append(row)
    return pd.DataFrame(rows).set_index("experiment")
