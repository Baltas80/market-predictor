import numpy as np
import pandas as pd
import pytest

from market_predictor.pipeline import run_final_lockbox_experiments
from market_predictor.session_calendar import session_table


def _data(n=180):
    rng = np.random.default_rng(11)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    index = session_table("2020-01-02", "2021-01-31")["close_utc"].iloc[:n]
    frame = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * (1 + rng.uniform(0, 0.01, n)),
            "low": close * (1 - rng.uniform(0, 0.01, n)),
            "close": close,
            "volume": rng.integers(1000, 10000, n),
            "macro_rate": rng.normal(3.0, 0.2, n),
            "geo_pressure": rng.normal(0.0, 1.0, n),
        },
        index=pd.DatetimeIndex(index),
    )
    return frame


def test_final_lockbox_runs_a_b_c_on_same_test_block():
    results = run_final_lockbox_experiments(
        _data(),
        horizon=5,
        test_fraction=0.2,
        macro_features=["macro_rate"],
        geopolitical_features=["geo_pressure"],
    )
    assert [result.name for result in results] == [
        "technical",
        "technical_macro",
        "technical_macro_geopolitical",
    ]
    assert all(len(result.evaluations) == 1 for result in results)
    indexes = [result.predictions.index for result in results]
    assert all(index.equals(indexes[0]) for index in indexes[1:])
    assert all(len(result.predictions) == len(indexes[0]) for result in results)


def test_final_lockbox_abc_rejects_missing_external_features():
    with pytest.raises(ValueError, match="Missing columns"):
        run_final_lockbox_experiments(
            _data(),
            macro_features=["does_not_exist"],
        )


def test_final_lockbox_abc_rejects_insufficient_data():
    with pytest.raises(ValueError, match="not enough observations"):
        run_final_lockbox_experiments(_data(30), horizon=20, test_fraction=0.4)
