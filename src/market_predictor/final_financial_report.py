"""Automatic final financial comparison report for fixed OOS predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .financial_matrix import evaluate_financial_matrix, period_stability
from .reproducibility import canonical_json_hash


@dataclass(frozen=True)
class FinancialReport:
    """Complete machine-readable final comparison package."""
    matrix: pd.DataFrame
    stability: pd.DataFrame
    result_hash: str


def _stability_rows(
    predictions: dict[str, pd.DataFrame],
    *,
    probability_column: str,
    threshold: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    periods: dict[str, tuple[str, str]],
    periods_per_year: int,
) -> pd.DataFrame:
    from .financial import backtest_long_only

    frames = []
    for name, prediction in predictions.items():
        frame = prediction[[probability_column, "close"]]
        backtest, _ = backtest_long_only(
            frame,
            probability_column=probability_column,
            threshold=threshold,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            periods_per_year=periods_per_year,
        )
        stability = period_stability(backtest, periods=periods, periods_per_year=periods_per_year)
        stability.insert(0, "experiment", name)
        frames.append(stability)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_final_financial_report(
    predictions: dict[str, pd.DataFrame],
    *,
    benchmark: pd.DataFrame,
    periods: dict[str, tuple[str, str]],
    probability_column: str = "prob_up",
    threshold: float = 0.5,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 0.0,
    periods_per_year: int = 252,
) -> FinancialReport:
    """Build matrix + temporal stability without retraining or OOS selection."""
    matrix = evaluate_financial_matrix(
        predictions,
        benchmark=benchmark,
        probability_column=probability_column,
        threshold=threshold,
        periods_per_year=periods_per_year,
    )
    stability = _stability_rows(
        predictions,
        probability_column=probability_column,
        threshold=threshold,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        periods=periods,
        periods_per_year=periods_per_year,
    )
    payload = {
        "matrix": matrix.to_dict(orient="records"),
        "stability": stability.to_dict(orient="records"),
    }
    return FinancialReport(matrix=matrix, stability=stability, result_hash=canonical_json_hash(payload))


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a simple dependency-free GitHub Markdown table."""
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_financial_report(report: FinancialReport, output_dir: str | Path) -> dict[str, Path]:
    """Write deterministic CSV/JSON/Markdown outputs and return their paths."""
    import json

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    matrix_path = directory / "financial_matrix.csv"
    stability_path = directory / "financial_stability.csv"
    json_path = directory / "financial_report.json"
    markdown_path = directory / "financial_report.md"
    report.matrix.to_csv(matrix_path, index=False)
    report.stability.to_csv(stability_path, index=False)
    payload = {
        "result_hash": report.result_hash,
        "matrix": report.matrix.to_dict(orient="records"),
        "stability": report.stability.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, sort_keys=True, default=str, indent=2) + "\n", encoding="utf-8")
    markdown = "# Final financial report\n\n"
    markdown += f"Result hash: `{report.result_hash}`\n\n"
    markdown += "## Matrix\n\n" + _markdown_table(report.matrix) + "\n\n"
    markdown += "## Temporal stability\n\n" + _markdown_table(report.stability) + "\n"
    markdown_path.write_text(markdown, encoding="utf-8")
    return {"matrix": matrix_path, "stability": stability_path, "json": json_path, "markdown": markdown_path}
