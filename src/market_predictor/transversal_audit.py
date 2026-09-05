"""One-call audit for the integrity gates used by OOS research."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .data_quality import quality_report
from .leakage import assert_event_availability_before_decision, assert_monotonic_unique_index, assert_target_is_future, audit_feature_columns


def run_transversal_audit(
    *,
    market: pd.DataFrame,
    features: pd.DataFrame,
    events: pd.DataFrame | None = None,
    purge_gap: int | None = None,
    horizon: int | None = None,
) -> dict[str, Any]:
    """Run quality, feature-name and point-in-time checks without modifying data.

    The audit returns a machine-readable report and raises on integrity failures.
    ``purge_gap`` is checked against ``horizon`` when both are supplied; callers
    should still use the fold-level purge assertion during model evaluation.
    """
    report = quality_report(market=market, events=events)
    if not report["passed"]:
        failed = [check["name"] for check in report["checks"] if not check["passed"]]
        raise ValueError(f"data-quality audit failed: {failed}")

    assert_monotonic_unique_index(market.index)
    if not isinstance(features.index, pd.DatetimeIndex):
        raise ValueError("Feature frame must use a DatetimeIndex")
    assert_monotonic_unique_index(features.index)

    suspicious = audit_feature_columns(list(features.columns))
    if suspicious:
        raise ValueError(f"suspicious feature columns require review: {suspicious}")

    if "decision_time" in features.columns and "target_end" in features.columns:
        assert_target_is_future(features)

    if events is not None and not events.empty:
        assert_event_availability_before_decision(events, decision_times=features.index)

    if purge_gap is not None:
        if purge_gap < 0:
            raise ValueError("purge_gap cannot be negative")
        if horizon is not None and purge_gap < horizon:
            raise ValueError("purge_gap must be at least the target horizon")

    return {
        "passed": True,
        "market_observations": len(market),
        "feature_observations": len(features),
        "event_observations": 0 if events is None else len(events),
        "suspicious_feature_columns": suspicious,
        "purge_gap": purge_gap,
        "horizon": horizon,
    }
