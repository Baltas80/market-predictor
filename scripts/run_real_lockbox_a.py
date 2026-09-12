"""Run the frozen lockbox using the validated GDELT 1.0 daily source."""
from __future__ import annotations

import runpy
from pathlib import Path

SOURCE_ID = "GDELT_1_Daily_Event_Database"


def main() -> None:
    """Delegate to the existing lockbox runner after the final PIT gate."""
    staging = Path("data/historical")
    missing = staging / "raw" / "events_gdelt_missing.csv"
    if missing.exists() and missing.stat().st_size:
        import pandas as pd
        frame = pd.read_csv(missing)
        if not frame.empty and "source_id" in frame.columns:
            unresolved = frame.loc[frame["source_id"] == SOURCE_ID]
            if not unresolved.empty:
                raise RuntimeError(f"GDELT 1.0 PIT gate refused: {len(unresolved)} source-days remain unresolved")
    runpy.run_path("scripts/run_real_lockbox.py", run_name="__main__")


if __name__ == "__main__":
    main()
