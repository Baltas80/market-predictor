import numpy as np
import pandas as pd

from market_predictor.pipeline import run_directional_baseline


def test_directional_baseline_uses_walk_forward_not_final_lockbox(monkeypatch):
    index = pd.date_range("2020-01-01", periods=80, tz="UTC", freq="D")
    close = np.linspace(100.0, 140.0, len(index))
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

    predictions = pd.DataFrame(
        {"actual": [1, 0], "prob_up": [0.9, 0.1]},
        index=index[-2:],
    )

    monkeypatch.setattr(
        "market_predictor.pipeline.run_baseline",
        lambda *args, **kwargs: (predictions, []),
    )

    _, directional_frame, metrics = run_directional_baseline(frame)
    assert directional_frame["position"].tolist() == [1, -1]
    assert metrics["long_signals"] == 1
    assert metrics["short_signals"] == 1
