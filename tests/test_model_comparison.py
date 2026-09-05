import numpy as np
import pandas as pd
import pytest

from market_predictor.model_comparison import ModelScore, compare_models, decide_challenger


def dataset(rows: int = 80):
    index = pd.date_range("2026-01-01", periods=rows, freq="D", tz="UTC")
    x = pd.DataFrame(
        {
            "momentum": np.linspace(-2, 2, rows),
            "volatility": np.sin(np.arange(rows) / 5.0),
            "macro": np.cos(np.arange(rows) / 7.0),
        },
        index=index,
    )
    y = ((x["momentum"] + 0.5 * x["macro"]) > 0).astype(int)
    return x.iloc[:60], y.iloc[:60], x.iloc[60:], y.iloc[60:]


def test_compare_models_uses_same_oos_index_and_returns_all_models():
    x_train, y_train, x_test, y_test = dataset()
    table, fitted = compare_models(x_train, y_train, x_test, y_test)
    assert set(table["name"]) == {"logistic", "random_forest", "hist_gradient_boosting"}
    assert set(fitted) == set(table["name"])
    assert list(table.columns) == ["name", "accuracy", "roc_auc", "brier"]
    assert table["brier"].notna().all()


def test_compare_models_rejects_misaligned_target():
    x_train, y_train, x_test, y_test = dataset()
    with pytest.raises(ValueError, match="indexes must match"):
        compare_models(x_train, y_train.reset_index(drop=True), x_test, y_test)


def test_compare_models_rejects_non_chronological_oos():
    x_train, y_train, x_test, y_test = dataset()
    x_test = x_test.iloc[::-1]
    y_test = y_test.loc[x_test.index]
    with pytest.raises(ValueError, match="OOS index"):
        compare_models(x_train, y_train, x_test, y_test)


def test_challenger_requires_accuracy_and_brier_not_to_regress():
    champion = ModelScore("logistic", 0.60, 0.65, 0.24)
    challenger = ModelScore("random_forest", 0.65, 0.70, 0.20)
    decision = decide_challenger(champion, challenger)
    assert decision.challenger_is_better is True
    assert "eligible" in decision.reason


def test_challenger_is_rejected_if_accuracy_regresses():
    champion = ModelScore("logistic", 0.70, 0.70, 0.20)
    challenger = ModelScore("random_forest", 0.69, 0.72, 0.19)
    decision = decide_challenger(champion, challenger)
    assert decision.challenger_is_better is False
    assert "rejected" in decision.reason


def test_unknown_model_rejected():
    x_train, y_train, x_test, y_test = dataset()
    with pytest.raises(ValueError, match="unknown model"):
        compare_models(x_train, y_train, x_test, y_test, model_names=("unknown",))
