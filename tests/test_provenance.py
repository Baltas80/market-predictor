import pandas as pd
import pytest

from market_predictor.provenance import dataframe_fingerprint, observation_trace, source_record


def test_source_record_carries_raw_and_normalized_hashes(tmp_path):
    raw = tmp_path / "source.csv"
    raw.write_text("date,value\n2024-01-01,100\n", encoding="utf-8")
    normalized = pd.DataFrame({"value": [100.0]}, index=pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")]))

    record = source_record(
        raw,
        source_id="fred:test",
        retrieval_method="fixture",
        retrieved_at="2024-01-02T00:00:00Z",
        normalized_frame=normalized,
        coverage_start="2024-01-01",
        coverage_end="2024-01-01",
        version="fixture-v1",
        pit_policy="realtime_start",
    )

    assert record["raw_sha256"] == record["sha256"]
    assert record["normalized_sha256"] == dataframe_fingerprint(normalized)
    assert record["pit_policy"] == "realtime_start"


def test_observation_trace_requires_complete_chain():
    with pytest.raises(ValueError, match="complete observation provenance"):
        observation_trace(
            source_id="fred:test",
            raw_sha256="raw",
            normalized_sha256="normalized",
            observation_id="obs-1",
            decision_time="",
            pit_policy="realtime_start",
        )


def test_observation_trace_preserves_decision_time_and_hashes():
    trace = observation_trace(
        source_id="fred:test",
        raw_sha256="raw-hash",
        normalized_sha256="normalized-hash",
        observation_id="DGS10:2024-01-01",
        decision_time="2024-02-02T21:00:00Z",
        pit_policy="realtime_start+session_lag_1d",
        source_version="v1",
    )
    assert trace["raw_sha256"] == "raw-hash"
    assert trace["normalized_sha256"] == "normalized-hash"
    assert trace["decision_time"] == "2024-02-02T21:00:00Z"
