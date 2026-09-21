import numpy as np
import pandas as pd

from market_predictor.multiclass_pipeline import run_multiclass_baseline


def test_multiclass_pipeline_returns_all_direction_probabilities(monkeypatch):
    index = pd.date_range("2020-01-01", periods=80, tz="UTC", freq="D")
    close = 100.0 + np.sin(np.arange(len(index)) / 2.0) * 5.0 + np.arange(len(index)) * 0.05
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(len(index), 1000.0),
        },
        index=index,
    )

    def fake_fit_predict(x_train, y_train, x_test):
        n = len(x_test)
        probs = pd.DataFrame(
            np.tile([[0.2, 0.2, 0.6]], (n, 1)),
            index=x_test.index,
            columns=["prob_down", "prob_flat", "prob_up"],
        )
        predicted = pd.Series(1, index=x_test.index, name="predicted_direction")
        return object(), probs, predicted

    monkeypatch.setattr(
        "market_predictor.multiclass_pipeline.fit_predict_multiclass",
        fake_fit_predict,
    )
    result, evaluations = run_multiclass_baseline(frame)
    assert not result.empty
    assert list(result.columns) == [
        "prob_down",
        "prob_flat",
        "prob_up",
        "actual_direction",
        "predicted_direction",
    ]
    assert len(evaluations) == 3
