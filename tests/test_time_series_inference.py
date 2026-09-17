import numpy as np
import pytest

from market_predictor.time_series_inference import (
    benjamini_hochberg,
    paired_block_bootstrap_difference,
    paired_block_swap_test,
)


def _sample():
    y = np.array([0, 1, 1, 0, 1, 0, 1, 1] * 20)
    a = np.array([0.45, 0.60, 0.55, 0.40, 0.52, 0.48, 0.58, 0.62] * 20)
    b = np.array([0.50, 0.65, 0.60, 0.45, 0.57, 0.53, 0.63, 0.67] * 20)
    return y, a, b


def test_block_bootstrap_is_reproducible_and_reports_effect_with_ci():
    y, a, b = _sample()
    first = paired_block_bootstrap_difference(y, a, b, metric="brier", block_length=8, n_bootstrap=500, seed=11)
    second = paired_block_bootstrap_difference(y, a, b, metric="brier", block_length=8, n_bootstrap=500, seed=11)
    assert first == second
    assert first["difference_b_minus_a"] < 0
    assert first["ci_low"] <= first["difference_b_minus_a"] <= first["ci_high"]
    assert first["bootstrap_replicates"] == 500


def test_block_swap_test_returns_valid_two_sided_p_value():
    y, a, b = _sample()
    result = paired_block_swap_test(y, a, b, metric="accuracy", block_length=8, n_permutations=500, seed=7)
    assert 0.0 < result["p_value_two_sided"] <= 1.0
    assert result["permutations"] == 500


def test_benjamini_hochberg_is_monotone_after_sorting():
    adjusted = benjamini_hochberg([0.01, 0.04, 0.02, 0.20])
    assert np.all((adjusted >= 0) & (adjusted <= 1))
    order = np.argsort([0.01, 0.04, 0.02, 0.20])
    assert np.all(np.diff(adjusted[order]) >= -1e-12)


def test_invalid_inputs_fail_closed():
    y, a, b = _sample()
    with pytest.raises(ValueError):
        paired_block_bootstrap_difference(y, a, b, block_length=0)
    with pytest.raises(ValueError):
        paired_block_bootstrap_difference(y, a, b, metric="unknown")
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1, np.nan])
