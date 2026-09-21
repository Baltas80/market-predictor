from pathlib import Path

import pandas as pd

import scripts.ingest_gdelt_chunk as ingest


def _checkpoint(path: Path, event_id: str, day: str) -> None:
    frame = pd.DataFrame(
        [
            {
                "event_id": event_id,
                "event_time": f"{day}T00:00:00+00:00",
                "published_at": "",
                "available_at": f"{pd.Timestamp(day).date()}T10:00:00+00:00",
                "source_id": ingest.GDELT_SOURCE_ID,
                "category": "political_crisis",
                "severity": 0.5,
                "country": "",
                "entity": "",
                "sector": "",
                "duration_days": 0.0,
                "media_intensity": 0.0,
                "surprise": 0.0,
                "checkpoint_version": ingest.GDELT_CHECKPOINT_VERSION,
            }
        ]
    )
    frame.to_csv(path, index=False)


def test_gdelt_stream_mode_does_not_concat_daily_frames(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    _checkpoint(checkpoint_dir / "day_2016-08-15.csv", "1", "2016-08-15")
    _checkpoint(checkpoint_dir / "day_2016-08-16.csv", "2", "2016-08-16")

    def fail_concat(*args, **kwargs):
        raise AssertionError("global pandas.concat must not run in streaming mode")

    monkeypatch.setattr(ingest.pd, "concat", fail_concat)

    frame, missing = ingest.fetch_gdelt_chunk(
        "2016-08-15",
        "2016-08-16",
        checkpoint_dir=checkpoint_dir,
        collect_frames=False,
    )

    assert frame.empty
    assert missing.empty
