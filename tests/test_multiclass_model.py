import pandas as pd
import pytest

from market_predictor.directional_target import DIRECTION_CLASSES
from market_predictor.multiclass_model import fit_predict_multiclass


def test_multiclass_model_exposes_down_flat_up_probabilities():
    x_train = pd.DataFrame({"x": [-2.0, -1.0, 1.0, 2.0, 0.0, 0.5]})
    y_train = pd.Series([-1, -1, 1, 1, 0, 0])
    x_test = pd.DataFrame({"x": [-1.5, 0.0, 1.5]})
    _, probabilities, predictions = fit_predict_multiclass(x_train, y_train, x_test)

    assert list(probabilities.columns) == ["prob_down", "prob_flat", "prob_up"]
    assert list(predictions.index) == list(x_test.index)
    assert set(predictions.unique()).issubset(set(DIRECTION_CLASSES))
    assert (probabilities.sum(axis=1) == pytest.approx(1.0)).all()


def test_multiclass_model_fails_closed_when_training_class_is_missing():
    x_train = pd.DataFrame({"x": [-1.0, 1.0]})
    y_train = pd.Series([-1, 1])
    x_test = pd.DataFrame({"x": [0.0]})
    with pytest.raises(ValueError):
        fit_predict_multiclass(x_train, y_train, x_test)
