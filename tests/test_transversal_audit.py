import pandas as pd
import pytest

from market_predictor.transversal_audit import run_transversal_audit


def _market():
    index = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": [100, 101, 102, 103], "high": [101, 102, 103, 104], "low": [99, 100, 101, 102], "close": [100, 101, 102, 103], "volume": [10, 11, 12, 13]},
        index=index,
    )


def test_audit_accepts_point_in_time_inputs():
    market = _market()
    features = pd.DataFrame({"return_1d": [0.01, 0.02, 0.01, 0.01]}, index=market.index)
    events = pd.DataFrame({"event_id": ["e1"], "published_at": [market.index[1]], "severity": [0.5]})
    report = run_transversal_audit(market=market, features=features, events=events, purge_gap=5, horizon=5)
    assert report["passed"] is True
    assert report["event_observations"] == 1


def test_audit_rejects_suspicious_feature_name():
    market = _market()
    features = pd.DataFrame({"future_return": [0.0] * 4}, index=market.index)
    with pytest.raises(ValueError, match="suspicious feature"):
        run_transversal_audit(market=market, features=features)


def test_audit_rejects_event_after_latest_decision():
    market = _market()
    features = pd.DataFrame({"return_1d": [0.0] * 4}, index=market.index)
    events = pd.DataFrame({"event_id": ["e1"], "published_at": [market.index[-1] + pd.Timedelta(days=1)], "severity": [0.5]})
    with pytest.raises(ValueError, match="unavailable"):
        run_transversal_audit(market=market, features=features, events=events)


def test_audit_rejects_insufficient_purge_gap():
    market = _market()
    features = pd.DataFrame({"return_1d": [0.0] * 4}, index=market.index)
    with pytest.raises(ValueError, match="purge_gap"):
        run_transversal_audit(market=market, features=features, purge_gap=2, horizon=5)
