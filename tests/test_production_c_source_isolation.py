from pathlib import Path

import pandas as pd

from scripts import run_real_lockbox


def test_production_c_loads_only_canonical_gdelt1(monkeypatch, tmp_path: Path):
    normalized = tmp_path / "normalized"
    normalized.mkdir(parents=True)
    gdelt_path = normalized / "events_gdelt.csv"
    gdelt_path.write_text("fixture", encoding="utf-8")

    calls = []

    def fake_loader(path):
        calls.append(Path(path).name)
        return [
            type(
                "Event",
                (),
                {
                    "available_at": pd.Timestamp("2025-01-01T11:00:00Z"),
                    "published_at": pd.NaT,
                    "event_time": pd.Timestamp("2025-01-01T00:00:00Z"),
                },
            )()
        ]

    monkeypatch.setattr(run_real_lockbox, "load_events_csv", fake_loader)

    events = run_real_lockbox._load_production_c_events(tmp_path)

    assert len(events) == 1
    assert calls == ["events_gdelt.csv"]
    assert not (normalized / "events_sec_litigation.csv").exists()


def test_production_c_fails_closed_when_gdelt1_file_missing(tmp_path: Path):
    try:
        run_real_lockbox._load_production_c_events(tmp_path)
    except RuntimeError as exc:
        assert "GDELT 1.0" in str(exc)
    else:
        raise AssertionError("production C must fail closed without GDELT 1.0 events")
