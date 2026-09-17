"""Run the frozen lockbox using the validated GDELT 1.0 daily source."""
from __future__ import annotations

import runpy
from pathlib import Path

import pandas as pd

from market_predictor.gdelt1 import GDELT_SOURCE_ID

SOURCE_ID = GDELT_SOURCE_ID


def main() -> None:
    """Delegate to the existing lockbox runner after the final PIT gate."""
    staging = Path("data/historical")
    missing = staging / "raw" / "events_gdelt_missing.csv"
    if missing.exists() and missing.stat().st_size:
        frame = pd.read_csv(missing)
        if not frame.empty and "source_id" in frame.columns:
            unresolved = frame.loc[frame["source_id"].astype(str) == SOURCE_ID]
            if not unresolved.empty:
                raise RuntimeError(f"GDELT 1.0 PIT gate refused: {len(unresolved)} source-days remain unresolved")
    runpy.run_path("scripts/run_real_lockbox.py", run_name="__main__")


if __name__ == "__main__":
    main()
