import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_real_lockbox import _assert_required_gdelt_provenance


def _write_provenance(staging: Path, *, source_id: str = "GDELT_1_Event_Database", policy: str = "GDELT 1.0 daily file publication boundary: conservative next-day 06:00 America/New_York publication boundary; original article publication time is unknown", missing_rows: list[dict] | None = None, chunk_count: int = 1, expected_chunk_count: int = 1) -> None:
    (staging / "raw").mkdir(parents=True, exist_ok=True)
    (staging / "normalized").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        missing_rows or [],
        columns=["date", "source_id", "status", "error_type", "error"],
    ).to_csv(staging / "raw" / "events_gdelt_missing.csv", index=False)
    (staging / "source_manifest.json").write_text(
        json.dumps(
            [
                {
                    "source_id": source_id,
                    "source_type": "events",
                    "coverage_start": "2015-02-19T00:00:00+00:00",
                    "coverage_end": "2025-12-31T00:00:00+00:00",
                    "rows": 1,
                    "sha256": "fixture",
                    "retrieval_version": "fixture",
                    "availability_policy": policy,
                }
            ]
        ),
        encoding="utf-8",
    )
    (staging / "staging_result.json").write_text(
        json.dumps(
            {
                "status": "admissible_with_source_limits",
                "gdelt_missing_day_count": 0,
                "gdelt_chunk_count": chunk_count,
                "gdelt_expected_chunk_count": expected_chunk_count,
            }
        ),
        encoding="utf-8",
    )


def test_lockbox_provenance_is_fail_closed_when_gap_manifest_missing(tmp_path: Path):
    with pytest.raises(RuntimeError, match="gap manifest is missing"):
        _assert_required_gdelt_provenance(tmp_path)


def test_lockbox_provenance_rejects_noncanonical_gdelt_source(tmp_path: Path):
    _write_provenance(tmp_path, source_id="GDELT_1_Daily_Event_Database")
    with pytest.raises(RuntimeError, match="non-canonical GDELT"):
        _assert_required_gdelt_provenance(tmp_path)


def test_lockbox_provenance_rejects_unresolved_required_gap(tmp_path: Path):
    _write_provenance(
        tmp_path,
        missing_rows=[
            {
                "date": "2025-01-02",
                "source_id": "GDELT_1_Event_Database",
                "status": "missing",
                "error_type": "Timeout",
                "error": "fixture",
            }
        ],
    )
    with pytest.raises(RuntimeError, match="source-days remain missing"):
        _assert_required_gdelt_provenance(tmp_path)


def test_lockbox_provenance_accepts_complete_canonical_staging(tmp_path: Path):
    _write_provenance(tmp_path)
    _assert_required_gdelt_provenance(tmp_path)


def test_lockbox_provenance_rejects_chunk_matrix_mismatch(tmp_path: Path):
    _write_provenance(tmp_path, chunk_count=1, expected_chunk_count=2)
    with pytest.raises(RuntimeError, match="chunk count"):
        _assert_required_gdelt_provenance(tmp_path)
