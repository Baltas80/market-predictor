"""Probability calibration diagnostics for out-of-sample predictions."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

def calibration_summary(y_true, probabilities, *, bins=10):
    """Return deterministic calibration bins, Brier score and ECE."""
    if bins < 2: raise ValueError("bins must be >= 2")
    y=pd.to_numeric(y_true,errors="coerce"); p=pd.to_numeric(probabilities,errors="coerce")
    valid=y.notna() & p.notna(); y=y.loc[valid].astype(int); p=p.loc[valid].clip(0,1)
    if len(y)==0: raise ValueError("No valid observations")
    edges=np.linspace(0,1,bins+1); rows=[]; ece=0.0
    for i in range(bins):
        mask=(p>=edges[i]) & (p<=edges[i+1] if i==bins-1 else p<edges[i+1]); count=int(mask.sum())
        if not count: continue
        mp=float(p.loc[mask].mean()); obs=float(y.loc[mask].mean()); ece += count/len(y)*abs(mp-obs)
        rows.append({"bin":i,"count":count,"mean_probability":mp,"observed_frequency":obs})
    return {"observations":len(y),"brier":float(brier_score_loss(y,p)),"ece":float(ece),"bins":pd.DataFrame(rows)}
