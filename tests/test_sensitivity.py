import numpy as np
import pandas as pd
import pytest

from market_predictor.sensitivity import run_sensitivity_grid


def _predictions(n=30):
    rng = np.random.default_rng(123)
    index = pd.date_range("2025-01-01", periods=n, tz="UTC")
    return pd.DataFrame(
        {
            "prob_up": rng.uniform(0.2, 0.8, n),
            "close": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n)),
        },
        index=index,
    )


def test_grid_has_expected_cardinality_and_parameters():
    result = run_sensitivity_grid(
        _predictions(),
        thresholds=[0.5, 0.6],
        transaction_costs_bps=[0.0, 5.0],
        slippages_bps=[0.0, 2.0, 5.0],
    )
    assert len(result) == 12
    assert set(result["threshold"]) == {0.5, 0.6}
    assert set(result["transaction_cost_bps"]) == {0.0, 5.0}
    assert set(result["slippage_bps"]) == {0.0, 2.0, 5.0}
    assert np.isfinite(result["total_return"]).all()


def test_higher_costs_cannot_improve_same_signal_path():
    predictions = _predictions()
    result = run_sensitivity_grid(
        predictions,
        thresholds=[0.55],
        transaction_costs_bps=[0.0, 10.0],
        slippages_bps=[0.0],
    ).sort_values("transaction_cost_bps")
    assert result.iloc[1]["total_return"] <= result.iloc[0]["total_return"] + 1e-12


def test_empty_grid_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        run_sensitivity_grid(_predictions(), thresholds=[])
