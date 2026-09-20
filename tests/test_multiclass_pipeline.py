import numpy as np
import pandas as pd
import pytest

from market_predictor.multiclass_pipeline import run_multiclass_baseline


def _frame(index):
    close = 100.0 + np.sin(np.arange(len(index)) / 2.0) * 5.0 + np.arange(len(index)) * 0.05
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(len(index), 1000.0),
        },
        index=index,
    )


def test_multiclass_pipeline_returns_probabilities_and_timestamp_audit(monkeypatch):
    index = pd.date_range("2020-01-01", periods=80, tz="UTC", freq="D")
    frame = _frame(index)

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
    assert {
        "prob_down", "prob_flat", "prob_up", "actual_direction",
        "predicted_direction", "fold_id", "train_start", "train_end",
        "test_start", "test_end", "purge_rule", "embargo", "embargo_rule",
    }.issubset(result.columns)
    assert len(evaluations) == 4
    assert (result["purge_rule"] == "label_end_time < test_start").all()
    assert (result["embargo_rule"] == "train_timestamp < test_start - embargo").all()


def test_multiclass_pipeline_purges_by_timestamp_on_irregular_index(monkeypatch):
    regular = pd.date_range("2020-01-01", periods=90, tz="UTC", freq="D")
    irregular = regular.delete([31, 32, 33, 60, 61])
    frame = _frame(irregular)

    def fake_fit_predict(x_train, y_train, x_test):
        n = len(x_test)
        probs = pd.DataFrame(
            np.tile([[0.3, 0.4, 0.3]], (n, 1)),
            index=x_test.index,
            columns=["prob_down", "prob_flat", "prob_up"],
        )
        predicted = pd.Series(0, index=x_test.index, name="predicted_direction")
        return object(), probs, predicted

    monkeypatch.setattr(
        "market_predictor.multiclass_pipeline.fit_predict_multiclass",
        fake_fit_predict,
    )
    result, _ = run_multiclass_baseline(frame)
    assert not result.empty
    for _, row in result.drop_duplicates("fold_id").iterrows():
        train_end = pd.Timestamp(row["train_end"])
        test_start = pd.Timestamp(row["test_start"])
        assert train_end < test_start - pd.Timedelta(days=1)


def test_multiclass_pipeline_rejects_unsorted_or_duplicate_timestamps():
    index = pd.date_range("2020-01-01", periods=20, tz="UTC", freq="D")
    frame = _frame(index)
    with pytest.raises(ValueError):
        run_multiclass_baseline(frame.sort_index(ascending=False))
    duplicate = pd.concat([frame.iloc[:6], frame.iloc[[5]], frame.iloc[6:]])
    with pytest.raises(ValueError):
        run_multiclass_baseline(duplicate)
