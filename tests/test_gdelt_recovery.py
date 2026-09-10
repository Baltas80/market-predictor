import pandas as pd

from scripts import recover_gdelt_gaps as recovery


def test_recover_day_maps_gdelt_export_id_and_pit_times(monkeypatch):
    raw = pd.DataFrame(
        {
            "global_event_id": ["g1"],
            "sql_date": [pd.Timestamp("2015-02-19", tz="UTC")],
            "date_added": [pd.Timestamp("2015-02-19 12:00", tz="UTC")],
            "category": ["political_crisis"],
            "severity": [0.5],
        }
    )
    monkeypatch.setattr(recovery, "load_gdelt_day", lambda day: raw.copy())

    frame, residual = recovery.recover_day("2015-02-19")

    assert residual is None
    assert frame is not None
    assert frame.loc[0, "event_id"] == "g1"
    assert frame.loc[0, "event_time"] == pd.Timestamp("2015-02-19", tz="UTC")
    assert frame.loc[0, "published_at"] == pd.Timestamp("2015-02-19 12:00", tz="UTC")
    assert frame.loc[0, "available_at"] == pd.Timestamp("2015-02-19 12:00", tz="UTC")


def test_recover_day_keeps_structural_failure_as_residual(monkeypatch):
    monkeypatch.setattr(
        recovery,
        "load_gdelt_day",
        lambda day: (_ for _ in ()).throw(ValueError("unsupported GDELT field count")),
    )
    monkeypatch.setattr(recovery.time, "sleep", lambda seconds: None)

    frame, residual = recovery.recover_day("2015-02-20")

    assert frame is None
    assert residual is not None
    assert residual["date"] == "2015-02-20"
    assert residual["error_type"] == "ValueError"
