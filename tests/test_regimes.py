import pandas as pd

from market_predictor.regimes import causal_regimes


def test_regime_labels_are_deterministic_and_causal_to_future_changes():
    idx = pd.date_range("2020-01-01", periods=700, freq="D", tz="UTC")
    close = pd.Series(100.0, index=idx)
    close.iloc[1:] = 100 * (1.0002 ** range(1, len(close)))
    base = causal_regimes(close)
    altered = close.copy()
    altered.iloc[-50:] *= 4.0
    changed = causal_regimes(altered)
    assert base.iloc[:-50].equals(changed.iloc[:-50])


def test_crash_is_labelled_crisis_only_after_information_is_available():
    idx = pd.date_range("2025-01-01", periods=40, tz="UTC")
    close = pd.Series(100.0, index=idx)
    close.iloc[20:] = 80.0
    labels = causal_regimes(close, volatility_window=3, threshold_window=5, crisis_return_window=5)
    assert labels.iloc[20] != "crisis"
    assert labels.iloc[-1] == "crisis"
