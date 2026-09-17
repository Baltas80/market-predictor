"""Run paired time-series inference on already-generated A/B OOS predictions.

This script is descriptive/inferential only. It never selects a model or
modifies the lockbox protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from market_predictor.time_series_inference import (
    benjamini_hochberg,
    paired_block_bootstrap_difference,
    paired_block_swap_test,
)


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "actual", "prob_up"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame[["date", "actual", "prob_up"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--technical", required=True)
    parser.add_argument("--technical-macro", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--block-lengths", default="5,20,60")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--permutation-reps", type=int, default=5000)
    parser.add_argument("--auc-bootstrap-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()

    a = _load(Path(args.technical))
    b = _load(Path(args.technical_macro))
    if not a["date"].equals(b["date"]):
        raise ValueError("A/B prediction files must share the exact same OOS timestamps")
    if not a["actual"].equals(b["actual"]):
        raise ValueError("A/B prediction files must share the exact same realized labels")

    block_lengths = [int(item) for item in args.block_lengths.split(",") if item.strip()]
    rows: list[dict[str, object]] = []
    for block_length in block_lengths:
        for metric in ("accuracy", "brier"):
            result = paired_block_bootstrap_difference(
                a["actual"], a["prob_up"], b["prob_up"],
                metric=metric, block_length=block_length,
                n_bootstrap=args.bootstrap_reps, seed=args.seed + block_length,
            )
            result.update(
                paired_block_swap_test(
                    a["actual"], a["prob_up"], b["prob_up"],
                    metric=metric, block_length=block_length,
                    n_permutations=args.permutation_reps, seed=args.seed + 1000 + block_length,
                )
            )
            rows.append(result)
        auc_result = paired_block_bootstrap_difference(
            a["actual"], a["prob_up"], b["prob_up"],
            metric="roc_auc", block_length=block_length,
            n_bootstrap=args.auc_bootstrap_reps, seed=args.seed + 2000 + block_length,
        )
        auc_perm = paired_block_swap_test(
            a["actual"], a["prob_up"], b["prob_up"],
            metric="roc_auc", block_length=block_length,
            n_permutations=args.permutation_reps, seed=args.seed + 3000 + block_length,
        )
        auc_result.update(auc_perm)
        rows.append(auc_result)

    result_frame = pd.DataFrame(rows)
    result_frame["bh_p_value"] = float("nan")
    for block_length in block_lengths:
        mask = result_frame["block_length"] == block_length
        result_frame.loc[mask, "bh_p_value"] = benjamini_hochberg(
            result_frame.loc[mask, "p_value_two_sided"].to_numpy(float)
        )
    result_frame.insert(0, "n_predictions", len(a))
    result_frame.insert(1, "oos_start", a["date"].iloc[0].isoformat())
    result_frame.insert(2, "oos_end", a["date"].iloc[-1].isoformat())
    result_frame.to_csv(args.output, index=False)

    metadata = {
        "n_predictions": len(a),
        "oos_start": a["date"].iloc[0].isoformat(),
        "oos_end": a["date"].iloc[-1].isoformat(),
        "block_lengths": block_lengths,
        "bootstrap_reps": args.bootstrap_reps,
        "auc_bootstrap_reps": args.auc_bootstrap_reps,
        "permutation_reps": args.permutation_reps,
        "seed": args.seed,
        "note": "Block lengths are a pre-declared sensitivity grid, not an optimization target. No lockbox selection is performed.",
    }
    output_path = Path(args.output)
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
