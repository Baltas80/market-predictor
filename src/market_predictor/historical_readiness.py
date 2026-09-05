"""Pre-ingestion readiness checks for point-in-time historical datasets."""
from __future__ import annotations
import pandas as pd
from .data_quality import quality_report
from .leakage import assert_event_availability_before_decision, assert_monotonic_unique_index

def audit_historical_inputs(*, market: pd.DataFrame, macro: pd.DataFrame | None = None, events: pd.DataFrame | None = None, decision_times: pd.DatetimeIndex | None = None) -> dict:
    """Return a strict machine-readable readiness report without repairing inputs."""
    report = quality_report(market=market, macro=macro, events=events)
    failures = [c["name"] for c in report["checks"] if not c["passed"]]
    try:
        assert_monotonic_unique_index(pd.DatetimeIndex(market.index))
    except (TypeError, ValueError) as exc:
        failures.append(f"market_index_integrity:{exc}")
    if events is not None and decision_times is not None:
        try:
            assert_event_availability_before_decision(events, decision_times=decision_times)
        except ValueError as exc:
            failures.append(f"event_availability:{exc}")
    return {"ready": not failures, "failures": failures, "quality": report}
