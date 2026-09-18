from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.benchmark_gdelt_parallel import _compare, _load_with_retries, _run_parallel


def _frame(day: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_event_id": ["2", "1"],
            "source_id": ["GDELT_1_Event_Database", "GDELT_1_Event_Database"],
            "value": [20, 10],
        }
    )


def test_parallel_results_are_deterministically_sorted():
    days = ["2025-01-02", "2025-01-01"]
    with patch("scripts.benchmark_gdelt_parallel.load_gdelt_day", side_effect=[_frame("2025-01-02"), _frame("2025-01-01")]):
        results = _run_parallel(days, workers=2)
    assert [item.date for item in results] == sorted(days)
    assert all(item.status == "ok" for item in results)


def test_comparison_requires_status_rows_and_hash_equality():
    slow = _load_with_retries
    del slow
    a = type("R", (), {"date": "2025-01-01", "status": "ok", "rows": 2, "sha256": "abc", "seconds": 1.0})()
    b = type("R", (), {"date": "2025-01-01", "status": "ok", "rows": 2, "sha256": "abc", "seconds": 1.0})()
    result = _compare([a], [b])
    assert result["all_days_identical"] is True
