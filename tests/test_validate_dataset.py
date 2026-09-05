from __future__ import annotations

import json
import subprocess
import sys


def test_validator_requires_an_input() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/validate_dataset.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_validator_writes_passing_report(tmp_path) -> None:
    market = tmp_path / "market.csv"
    report = tmp_path / "report.json"
    market.write_text(
        "date,open,high,low,close,volume\n"
        "2020-01-01,10,12,9,11,100\n"
        "2020-01-02,11,13,10,12,120\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "scripts/validate_dataset.py", "--market", str(market), "--report", str(report)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["passed"] is True


def test_validator_fails_on_bad_market(tmp_path) -> None:
    market = tmp_path / "market.csv"
    market.write_text(
        "date,open,high,low,close,volume\n"
        "2020-01-01,15,12,9,11,100\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "scripts/validate_dataset.py", "--market", str(market), "--report", str(tmp_path / "report.json")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
