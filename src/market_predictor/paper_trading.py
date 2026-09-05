"""No-capital signal contract kept separate from financial execution."""
from __future__ import annotations
from dataclasses import asdict,dataclass
from datetime import datetime
@dataclass(frozen=True)
class PaperSignal:
    signal_id:str; decision_time:datetime; probability_up:float; confidence:float; position:int; expected_risk:float
    def validate(self)->None:
        if not self.signal_id: raise ValueError("signal_id is required")
        if self.decision_time.tzinfo is None: raise ValueError("decision_time must be timezone-aware")
        if not 0<=self.probability_up<=1: raise ValueError("probability_up must be in [0, 1]")
        if not 0<=self.confidence<=1: raise ValueError("confidence must be in [0, 1]")
        if self.position not in (-1,0,1): raise ValueError("position must be -1, 0 or 1")
        if self.expected_risk<0: raise ValueError("expected_risk cannot be negative")
    def as_dict(self): self.validate(); return asdict(self)
