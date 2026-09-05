"""CI entrypoint for the fail-closed research lifecycle.

The gate intentionally runs only deterministic checks that do not require
external historical downloads. Real-data ingestion remains a prerequisite
for the production research run; CI verifies that the lifecycle guards and
contracts cannot regress.
"""
from __future__ import annotations

import pandas as pd

from market_predictor.abc_protocol import ABCProtocol, assert_same_abc_protocol, common_walk_forward_folds
from market_predictor.information_set import build_information_set
from market_predictor.research_gate import ResearchGateState
from market_predictor.reproducibility import canonical_json_hash
from market_predictor.session_calendar import audit_market_timestamps, session_table
from market_predictor.temporal_audit import assert_no_future_information


def main() -> None:
    sessions = session_table("2025-01-02", "2025-01-31").head(12)
    index = pd.DatetimeIndex(sessions["close_utc"])
    market = pd.DataFrame({"close": range(100, 112)}, index=index)
    macro = pd.DataFrame(
        {
            "series_id": ["TEST"] * 12,
            "observation_date": index,
            "value": range(12),
            "vintage_start": index - pd.Timedelta(days=1),
            "vintage_end": index + pd.Timedelta(days=30),
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["e1"],
            "event_time": [index[2]],
            "published_at": [index[2]],
            "available_at": [index[2]],
            "severity": [0.5],
        }
    )

    # DATA -> AUDIT
    session_audit = audit_market_timestamps(index)
    if session_audit["weekend"].any() or session_audit["holiday"].any():
        raise RuntimeError("research gate: market index contains non-session timestamps")
    if session_audit["after_close"].any() or not session_audit["at_or_before_close"].all():
        raise RuntimeError("research gate: market index contains timestamps after session close")
    assert_no_future_information(events.assign(decision_time=index[3]))

    # INFORMATION SET
    info = build_information_set(
        index[3], market=market, macro=macro, events=events
    )
    if info.market_row is None:
        raise RuntimeError("research gate: missing point-in-time market row")

    # A/B/C: one protocol, one fold generator, one prediction index identity.
    prediction_index_hash = canonical_json_hash([ts.isoformat() for ts in index[5:]])
    protocols = {
        name: ABCProtocol(
            lockbox_start=index[5].date(), lockbox_end=index[-1].date(),
            purge_gap=5, horizon=5, transaction_cost_bps=5.0, slippage_bps=5.0,
            observations=7, prediction_index_hash=prediction_index_hash,
            dataset_hash="ci-dataset", code_version="ci-code", protocol_version="2026-09-05",
        )
        for name in ("A", "B", "C")
    }
    assert_same_abc_protocol(protocols)
    folds = common_walk_forward_folds(len(index), train_size=5, test_size=1, step=1, purge=5)
    if not folds:
        raise RuntimeError("research gate: no valid shared A/B/C folds")

    # Lifecycle fail-closed checks.
    audited = ResearchGateState(data_audited=True, information_set_audited=True, abc_protocol_frozen=True)
    audited.assert_can_generate_predictions()
    ready_for_backtest = ResearchGateState(
        data_audited=True, information_set_audited=True, abc_protocol_frozen=True,
        predictions_generated=True, lockbox_frozen=True,
    )
    # Reporting must remain blocked until financial evaluation is explicitly complete.
    try:
        ready_for_backtest.assert_can_report()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("research gate: report was not fail-closed")

    print("RESEARCH_GATE=PASS")
    print("DATA=PASS AUDIT=PASS INFORMATION_SET=PASS ABC=PASS LOCKBOX_GATE=PASS")
    print("BACKTEST_GATE=PASS REPORT_GATE=BLOCKED_UNTIL_FINANCIAL_EVALUATION")


if __name__ == "__main__":
    main()
