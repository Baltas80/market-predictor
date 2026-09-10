"""Deterministic, paper-only decision layer for validated model forecasts.

The engine is deliberately fail-closed: a directional action is emitted only
when data, model, calibration, lockbox, regime, confidence, edge and risk gates
all pass. It never submits orders.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from math import isfinite
from typing import Literal

Action = Literal["LONG", "SHORT", "FLAT", "NO_TRADE"]
DataStatus = Literal["OK", "STALE", "INCOMPLETE", "INVALID"]


@dataclass(frozen=True)
class DecisionConfig:
    """Immutable decision thresholds; these must not be tuned on the lockbox."""

    long_threshold: float = 0.55
    short_threshold: float = 0.45
    min_confidence: float = 0.55
    safety_margin_bps: float = 1.0
    max_signal_age_seconds: int = 86_400
    max_proposed_exposure: float = 1.0
    max_turnover: float = 0.25
    max_drawdown: float = 0.10

    def __post_init__(self) -> None:
        if not 0.5 <= self.long_threshold <= 1:
            raise ValueError("long_threshold must be in [0.5, 1]")
        if not 0 <= self.short_threshold <= 0.5:
            raise ValueError("short_threshold must be in [0, 0.5]")
        if self.short_threshold >= self.long_threshold:
            raise ValueError("short_threshold must be below long_threshold")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        for name in ("safety_margin_bps", "max_proposed_exposure", "max_turnover", "max_drawdown"):
            if getattr(self, name) < 0 or not isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_signal_age_seconds < 0:
            raise ValueError("max_signal_age_seconds must be non-negative")


@dataclass(frozen=True)
class PredictionSnapshot:
    timestamp: datetime
    symbol: str
    model_version: str
    feature_schema_version: str
    experiment_id: str
    probability: float
    confidence: float
    expected_edge_bps: float
    estimated_cost_bps: float
    slippage_bps: float
    signal_age_seconds: int
    data_status: DataStatus
    model_approved: bool
    calibration_available: bool
    lockbox_released: bool
    regime_valid: bool
    current_exposure: float = 0.0
    proposed_exposure: float = 0.0
    proposed_turnover: float = 0.0
    drawdown: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if not self.symbol or not self.model_version or not self.feature_schema_version or not self.experiment_id:
            raise ValueError("identity fields must be non-empty")
        for name in ("probability", "confidence"):
            value = getattr(self, name)
            if not 0 <= value <= 1 or not isfinite(value):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in ("expected_edge_bps", "estimated_cost_bps", "slippage_bps", "current_exposure", "proposed_exposure", "proposed_turnover", "drawdown"):
            if not isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.signal_age_seconds < 0:
            raise ValueError("signal_age_seconds must be non-negative")


@dataclass(frozen=True)
class Decision:
    contract_version: str
    timestamp: str
    symbol: str
    model_version: str
    feature_schema_version: str
    experiment_id: str
    probability: float
    confidence: float
    expected_edge_bps: float
    cost_bps: float
    slippage_bps: float
    edge_after_costs_bps: float
    action: Action
    reason_codes: tuple[str, ...]
    proposed_exposure: float
    current_exposure: float
    data_status: DataStatus

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["reason_codes"] = list(self.reason_codes)
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def decide(snapshot: PredictionSnapshot, config: DecisionConfig = DecisionConfig()) -> Decision:
    """Return one deterministic paper decision; all safety gates fail closed."""
    reasons: list[str] = []
    edge_after_costs = (
        snapshot.expected_edge_bps
        - snapshot.estimated_cost_bps
        - snapshot.slippage_bps
        - config.safety_margin_bps
    )

    if snapshot.data_status != "OK":
        reasons.append(f"DATA_{snapshot.data_status}")
    if snapshot.signal_age_seconds > config.max_signal_age_seconds and "DATA_STALE" not in reasons:
        reasons.append("DATA_STALE")
    if not snapshot.model_approved:
        reasons.append("MODEL_NOT_APPROVED")
    if not snapshot.calibration_available:
        reasons.append("CALIBRATION_UNAVAILABLE")
    if not snapshot.lockbox_released:
        reasons.append("LOCKBOX_NOT_RELEASED")
    if not snapshot.regime_valid:
        reasons.append("REGIME_OUTSIDE_VALIDATED")
    if snapshot.confidence < config.min_confidence:
        reasons.append("CONFIDENCE_BELOW_MIN")
    if edge_after_costs <= 0:
        reasons.append("EDGE_AFTER_COSTS_NON_POSITIVE")
    if snapshot.proposed_exposure < 0 or snapshot.proposed_exposure > config.max_proposed_exposure:
        reasons.append("EXPOSURE_LIMIT")
    if snapshot.proposed_turnover < 0 or snapshot.proposed_turnover > config.max_turnover:
        reasons.append("TURNOVER_LIMIT")
    if snapshot.drawdown < 0 or snapshot.drawdown > config.max_drawdown:
        reasons.append("DRAWDOWN_LIMIT")

    action: Action = "NO_TRADE"
    if not reasons:
        if snapshot.probability >= config.long_threshold:
            action = "LONG"
            reasons.append("LONG_THRESHOLD")
        elif snapshot.probability <= config.short_threshold:
            action = "SHORT"
            reasons.append("SHORT_THRESHOLD")
        else:
            action = "FLAT"
            reasons.append("NO_DIRECTIONAL_EDGE")

    return Decision(
        contract_version="decision-v1",
        timestamp=snapshot.timestamp.astimezone(timezone.utc).isoformat(),
        symbol=snapshot.symbol,
        model_version=snapshot.model_version,
        feature_schema_version=snapshot.feature_schema_version,
        experiment_id=snapshot.experiment_id,
        probability=snapshot.probability,
        confidence=snapshot.confidence,
        expected_edge_bps=snapshot.expected_edge_bps,
        cost_bps=snapshot.estimated_cost_bps,
        slippage_bps=snapshot.slippage_bps,
        edge_after_costs_bps=edge_after_costs,
        action=action,
        reason_codes=tuple(reasons),
        proposed_exposure=snapshot.proposed_exposure,
        current_exposure=snapshot.current_exposure,
        data_status=snapshot.data_status,
    )
