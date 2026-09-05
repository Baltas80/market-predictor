"""Experiment D=C+AI contract and evaluation plan."""
from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class DExperimentSpec:
    base_experiment:str="C"
    ai_overlay:str="AI"
    name:str="D"
    same_folds:bool=True
    same_purge_gap:bool=True
    same_lockbox:bool=True
    ai_is_incremental:bool=True
    quantitative_target_unchanged:bool=True
    def validate(self):
        if self.base_experiment!="C" or self.name!="D": raise ValueError("D must be defined as C + AI")
        if not all((self.same_folds,self.same_purge_gap,self.same_lockbox,self.ai_is_incremental,self.quantitative_target_unchanged)):
            raise ValueError("D violates the common evaluation protocol")

def build_d_protocol()->dict[str,object]:
    spec=DExperimentSpec(); spec.validate()
    return {"experiment":"D","definition":"C + AI overlay","same_folds":True,"same_purge_gap":True,"same_lockbox":True,"incremental_ai":True,"target":"unchanged quantitative future-return target","selection":"AI may not tune or rewrite the final lockbox"}
