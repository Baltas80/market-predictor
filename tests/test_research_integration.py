import pandas as pd
import pytest
from datetime import date, datetime, timezone

from market_predictor.calibration import calibration_summary
from market_predictor.events_audit import assert_events_available_by_decision, admitted_events_for_decision
from market_predictor.experiment_d import build_d_protocol
from market_predictor.historical_readiness import validate_historical_frame
from market_predictor.lockbox_manifest import LockboxManifest
from market_predictor.paper_trading import PaperSignal
from market_predictor.regimes import causal_regimes
from market_predictor.statistics import summarize_predictions


def test_statistics_and_calibration_contracts():
    p = pd.DataFrame({"actual": [0,1,1,0], "prob_up": [0.1,0.8,0.7,0.2]})
    s = summarize_predictions(p)
    c = calibration_summary(p.actual, p.prob_up, bins=4)
    assert s.observations == 4 and c["observations"] == 4


def test_event_audit_is_per_decision():
    events = pd.DataFrame({"published_at": pd.to_datetime(["2025-01-01", "2025-01-03"], utc=True),
                           "decision_time": pd.to_datetime(["2025-01-02", "2025-01-02"], utc=True)})
    with pytest.raises(ValueError):
        assert_events_available_by_decision(events)
    admitted = admitted_events_for_decision(events, datetime(2025,1,2,tzinfo=timezone.utc))
    assert len(admitted) == 1


def test_d_protocol_and_lockbox_manifest_are_deterministic():
    assert build_d_protocol()["definition"] == "C + AI overlay"
    manifest = LockboxManifest(date(2025,1,1), date(2025,12,31), 5, "data", "code", "v1", 5, 2, (("A", "hash"),))
    assert manifest.fingerprint() == manifest.fingerprint()


def test_paper_signal_requires_timezone():
    signal = PaperSignal("s1", datetime(2025,1,1,tzinfo=timezone.utc), .7, .8, 1, .1)
    signal.validate()


def test_causal_regimes_do_not_use_future_values():
    idx = pd.date_range("2020-01-01", periods=300, tz="UTC")
    close = pd.Series(range(100,400), index=idx, dtype=float)
    before = causal_regimes(close)
    changed = close.copy(); changed.iloc[-1] *= 100
    after = causal_regimes(changed)
    assert before.iloc[:-1].equals(after.iloc[:-1])


def test_historical_readiness_rejects_unsorted_data():
    idx = pd.to_datetime(["2025-01-02", "2025-01-01"], utc=True)
    with pytest.raises(ValueError):
        validate_historical_frame(pd.DataFrame({"close": [2.0,1.0]}, index=idx), required_columns=("close",))
