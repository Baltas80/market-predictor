from pathlib import Path

from scripts.ingest_historical import build_plan


def test_ingestion_plan_declares_fail_closed_stages(tmp_path: Path):
    plan = build_plan(tmp_path)
    stages = plan["stages"]

    assert [stage["name"] for stage in stages] == [
        "download",
        "raw",
        "hash",
        "normalize",
        "pit",
        "historical_gate",
        "dataset",
    ]
    assert all(stage["status"] == "blocked" for stage in stages)
    assert plan["execution_policy"]["historical_gate_required"] is True
    assert plan["execution_policy"]["no_lockbox_admission_from_download_alone"] is True


def test_ingestion_plan_has_distinct_raw_and_normalized_locations(tmp_path: Path):
    plan = build_plan(tmp_path)
    assert plan["raw_directory"] != plan["normalized_directory"]
    assert plan["manifest"] == str(tmp_path / "source_manifest.json")
