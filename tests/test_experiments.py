import numpy as np
import pandas as pd
import pytest

from market_predictor.backtest import Fold
from market_predictor.experiments import run_feature_ablation


def _data(n=80):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "return_1d": rng.normal(size=n),
        "return_5d": rng.normal(size=n),
        "volatility_20d": rng.random(n),
        "price_to_sma20": rng.normal(size=n),
        "volume_change": rng.normal(size=n),
        "range_pct": rng.random(n),
        "target": np.arange(n) % 2,
        "macro": rng.normal(size=n),
        "geo": rng.normal(size=n),
    })


def test_feature_ablation_uses_same_folds():
    data = _data()
    folds = [Fold(0, 50, 50, 65), Fold(0, 65, 65, 80)]
    results = run_feature_ablation(
        data, folds, macro_features=["macro"], geopolitical_features=["geo"]
    )
    assert [r.name for r in results] == [
        "technical", "technical_macro", "technical_macro_geopolitical"
    ]
    assert all(len(r.predictions) == 30 for r in results)


def test_missing_feature_is_rejected():
    data = _data().drop(columns=["geo"])
    folds = [Fold(0, 50, 50, 65)]
    with pytest.raises(ValueError, match="Missing columns"):
        run_feature_ablation(
            data, folds, macro_features=["macro"], geopolitical_features=["geo"]
        )
