"""Append-only paper-trading audit records; no execution is performed."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

@dataclass(frozen=True)
class PaperOutcome:
    signal_id: str
    decision_time: datetime
    probability_up: float
    position: int
    expected_risk: float
    outcome_time: datetime | None = None
    realized_return: float | None = None

    def validate(self) -> None:
        if not self.signal_id: raise ValueError("signal_id is required")
        if self.decision_time.tzinfo is None: raise ValueError("decision_time must be timezone-aware")
        if not 0 <= self.probability_up <= 1: raise ValueError("probability_up must be in [0,1]")
        if self.position not in (-1, 0, 1): raise ValueError("position must be -1, 0 or 1")
        if self.expected_risk < 0: raise ValueError("expected_risk cannot be negative")
        if self.outcome_time is not None:
            if self.outcome_time.tzinfo is None: raise ValueError("outcome_time must be timezone-aware")
            if self.outcome_time <= self.decision_time: raise ValueError("outcome_time must follow decision_time")

    def as_dict(self) -> dict[str, Any]:
        self.validate(); return asdict(self)

class PaperLedger:
    """In-memory append-only ledger suitable for deterministic audit tests."""
    def __init__(self) -> None: self._records: list[PaperOutcome] = []
    def append(self, record: PaperOutcome) -> None:
        record.validate()
        if any(r.signal_id == record.signal_id for r in self._records): raise ValueError("signal_id already exists")
        self._records.append(record)
    def records(self) -> tuple[PaperOutcome, ...]: return tuple(self._records)
