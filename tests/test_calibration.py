import pandas as pd
import pytest

from market_predictor.calibration import calibration_summary


def test_calibration_summary_includes_brier_ece_and_bins():
    y_true = pd.Series([0, 1, 1, 0])
    probabilities = pd.Series([0.1, 0.9, 0.9, 0.1])
    result = calibration_summary(y_true, probabilities, bins=2)
    assert result["observations"] == 4
    assert result["brier"] == pytest.approx(0.01)
    assert result["ece"] == pytest.approx(0.1)
    assert result["bins"]["count"].tolist() == [2, 2]


def test_calibration_places_probability_one_in_last_bin():
    y_true = pd.Series([1, 0])
    probabilities = pd.Series([1.0, 0.0])
    result = calibration_summary(y_true, probabilities, bins=2)
    bins = result["bins"]
    assert bins["count"].sum() == 2
    assert bins.iloc[-1]["mean_probability"] == pytest.approx(1.0)


def test_calibration_rejects_invalid_bin_count_and_empty_input():
    with pytest.raises(ValueError, match="bins"):
        calibration_summary(pd.Series([1]), pd.Series([0.5]), bins=1)
    with pytest.raises(ValueError, match="No valid"):
        calibration_summary(pd.Series([float("nan")]), pd.Series([float("nan")]))
