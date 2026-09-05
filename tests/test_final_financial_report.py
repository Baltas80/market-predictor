import pandas as pd

from market_predictor.final_financial_report import build_final_financial_report, write_financial_report


def test_final_report_contains_matrix_stability_and_hash(tmp_path):
    idx = pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC")
    close = pd.Series(range(100, 120), index=idx, dtype=float)
    predictions = {
        name: pd.DataFrame({"prob_up": 0.6, "close": close}, index=idx)
        for name in ("A", "B", "C")
    }
    report = build_final_financial_report(
        predictions,
        benchmark=pd.DataFrame({"close": close}, index=idx),
        periods={"first": ("2025-01-01", "2025-01-10"), "second": ("2025-01-11", "2025-01-20")},
    )
    assert len(report.matrix) == 4 * 3 * 2
    assert set(report.matrix.experiment) == {"A", "B", "C", "random"}
    assert set(report.stability.experiment) == {"A", "B", "C"}
    assert len(report.result_hash) == 64
    paths = write_financial_report(report, tmp_path)
    assert all(path.exists() for path in paths.values())
