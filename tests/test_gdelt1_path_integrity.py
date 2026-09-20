from pathlib import Path


def test_obsolete_gdelt1_paths_are_retired():
    assert not Path("scripts/ingest_gdelt_daily_chunk.py").exists()
    assert not Path("scripts/recover_gdelt_daily_gaps.py").exists()
    assert not Path("scripts/run_real_lockbox_a.py").exists()
    assert not Path(".github/workflows/real-lockbox-gdelt1.yml").exists()


def test_active_lockbox_workflow_uses_canonical_recovery_path():
    workflow = Path(".github/workflows/real-lockbox-backtest.yml").read_text(encoding="utf-8")
    assert "scripts/ingest_gdelt_chunk.py" in workflow
    assert "scripts/recover_gdelt_daily_gaps.py" not in workflow
    assert "scripts/ingest_gdelt_daily_chunk.py" not in workflow
    assert "scripts/run_real_lockbox_a.py" not in workflow


def test_repository_source_manifest_uses_only_canonical_gdelt1_identity():
    manifest = Path("data/source_manifest.yml").read_text(encoding="utf-8")
    assert "GDELT_1_Event_Database" in manifest
    assert "GDELT_1_Daily_Event_Database" not in manifest
