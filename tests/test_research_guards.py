import pandas as pd
import pytest
from datetime import date

from market_predictor.abc_protocol import ABCProtocol, SharedExperimentProtocol, assert_same_abc_protocol
from market_predictor.abc_execution_guard import build_abc_execution_plan
from market_predictor.backtest import Fold
from market_predictor.information_set import build_information_set
from market_predictor.research_gate import ResearchGateState
from market_predictor.temporal_audit import assert_no_future_information


def protocol(**overrides):
    values = dict(
        lockbox_start=date(2025, 1, 1), lockbox_end=date(2025, 12, 31),
        purge_gap=5, horizon=5, transaction_cost_bps=5.0,
        slippage_bps=0.0, observations=10, prediction_index_hash="idx",
        dataset_hash="data", code_version="code", protocol_version="abc-v1",
    )
    values.update(overrides)
    return ABCProtocol(**values)


def test_legacy_protocol_is_the_same_class():
    assert SharedExperimentProtocol is ABCProtocol


def test_abc_requires_exactly_one_shared_contract():
    p = protocol()
    assert_same_abc_protocol({"A": p, "B": p, "C": p})
    with pytest.raises(ValueError):
        assert_same_abc_protocol({"A": p, "B": protocol(horizon=6), "C": p})


def test_execution_plan_rejects_wrong_observation_count_and_purge():
    idx = pd.date_range("2025-01-01", periods=10, tz="UTC")
    fold = Fold(0, 3, 8, 10)
    with pytest.raises(ValueError, match="observations"):
        build_abc_execution_plan(protocol=protocol(observations=9), folds=[fold], observation_index=idx)
    with pytest.raises(ValueError, match="purge"):
        build_abc_execution_plan(protocol=protocol(), folds=[Fold(0, 6, 8, 10)], observation_index=idx)


def test_information_set_rejects_future_macro_and_events():
    decision = pd.Timestamp("2025-01-02 16:00", tz="UTC")
    market = pd.DataFrame({"close": [100.0]}, index=pd.DatetimeIndex([decision]))
    macro = pd.DataFrame({"vintage_start": pd.to_datetime(["2025-01-03"], utc=True), "value": [2.0]})
    events = pd.DataFrame({"available_at": pd.to_datetime(["2025-01-03"], utc=True)})
    info = build_information_set(decision_time=decision, market=market, macro=macro, events=events)
    assert info.macro_rows.empty and info.event_rows.empty


def test_temporal_gate_rejects_future_and_weekend_rows():
    frame = pd.DataFrame({
        "decision_time": pd.to_datetime(["2025-01-03", "2025-01-04"], utc=True),
        "available_at": pd.to_datetime(["2025-01-02", "2025-01-04"], utc=True),
    })
    with pytest.raises(ValueError, match="future information|weekend"):
        assert_no_future_information(frame)


def test_research_gate_is_fail_closed():
    state = ResearchGateState()
    with pytest.raises(RuntimeError):
        state.assert_can_generate_predictions()
    with pytest.raises(RuntimeError):
        state.assert_can_backtest()
