"""Fail-closed gates for the research lifecycle."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from market_predictor.session_calendar import audit_market_timestamps


@dataclass(frozen=True)
class ResearchGateState:
    """Immutable lifecycle state for data-to-report execution."""

    data_audited: bool = False
    information_set_audited: bool = False
    abc_protocol_frozen: bool = False
    predictions_generated: bool = False
    lockbox_frozen: bool = False
    financial_evaluation_complete: bool = False

    def assert_can_generate_predictions(self) -> None:
        if not (self.data_audited and self.information_set_audited and self.abc_protocol_frozen):
            raise RuntimeError("predictions blocked: data, information set and A/B/C protocol must be audited/frozen")

    def assert_can_backtest(self) -> None:
        self.assert_can_generate_predictions()
        if not self.predictions_generated or not self.lockbox_frozen:
            raise RuntimeError("financial backtest blocked: predictions and lockbox must be frozen")

    def assert_can_report(self) -> None:
        self.assert_can_backtest()
        if not self.financial_evaluation_complete:
            raise RuntimeError("final report blocked: financial evaluation is incomplete")


def assert_market_session_audit(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Fail closed on non-session dates and timestamps after the applicable close."""
    audit = audit_market_timestamps(index)
    invalid = audit.loc[(~audit["valid_market_day"]) | audit["after_close"]]
    if not invalid.empty:
        raise RuntimeError(
            "market session audit failed: timestamps fall on a non-session day "
            "or after the applicable market close"
        )
    return audit
