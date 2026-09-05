from datetime import datetime, timezone

import pandas as pd
import pytest

from market_predictor.experiments import ExperimentResult
from market_predictor.paper_abc import signals_from_predictions, simulate_abc_paper_trading


UTC = timezone.utc


def test_signal_adapter_declares_positions_from_probability():
    index = pd.DatetimeIndex(
        [datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)]
    )
    predictions = pd.DataFrame({"prob_up": [0.8, 0.5]}, index=index)
    signals = signals_from_predictions(predictions)
    assert [s.position for s in signals] == [1, 0]
    assert signals[0].confidence == pytest.approx(0.6)


def test_abc_adapter_requires_exact_shared_index():
    index = pd.date_range("2026-01-01", periods=2, tz="UTC")
    predictions = pd.DataFrame({"prob_up": [0.8, 0.2]}, index=index)
    result_a = ExperimentResult("technical", (), [], predictions)
    result_b = ExperimentResult("technical_macro", (), [], predictions.copy())
    prices = pd.Series([100.0, 105.0], index=index)
    output = simulate_abc_paper_trading([result_a, result_b], prices)
    assert set(output) == {"technical", "technical_macro"}
    assert output["technical"][0] > 100_000


def test_abc_adapter_rejects_different_price_index():
    index = pd.date_range("2026-01-01", periods=2, tz="UTC")
    predictions = pd.DataFrame({"prob_up": [0.8, 0.2]}, index=index)
    result = ExperimentResult("technical", (), [], predictions)
    prices = pd.Series([100.0, 105.0], index=index.shift(1, freq="D"))
    with pytest.raises(ValueError, match="exact shared OOS prediction index"):
        simulate_abc_paper_trading([result], prices)
