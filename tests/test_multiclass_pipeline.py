import numpy as np
import pandas as pd

from market_predictor.multiclass_pipeline import run_multiclass_baseline


def test_multiclass_pipeline_returns_all_direction_probabilities(monkeypatch):
    index = pd.date_range("2020-01-01", periods=40, tz="UTC", freq="D")
    close = np.linspace(100.0, 120.0, len(index))
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(len(index), 1000.0),
        },
        index=index,
    )

    def fake_runner(*args, **kwargs):
        result = pd.DataFrame(
            {
                "prob_down": [0.8, 0.1, 0.2],
                "prob_flat": [0.1, 0.2, 0.2],
                "prob_up": [0.1, 0.7, 0.6],
                "actual_direction": [-1, 0, 1],
                "predicted_direction": [-1, 1, 1],
            },
            index=index[-3:],
        )
        return result, []

    monkeypatch.setattr(
        "market_predictor.multiclass_pipeline.make_walk_forward_folds",
        lambda *args, **kwargs: [],
    )
    result, evaluations = run_multiclass_baseline(frame)
    assert evaluations == []
    assert result.empty
