from datetime import datetime, timezone

import pytest

from market_predictor.paper_trading import PaperSignal


def test_paper_signal_validates_and_serializes():
    signal = PaperSignal("s1", datetime(2026, 1, 1, tzinfo=timezone.utc), 0.7, 0.8, 1, 0.03)
    assert signal.as_dict()["signal_id"] == "s1"


def test_paper_signal_rejects_naive_time_and_invalid_probability():
    with pytest.raises(ValueError):
        PaperSignal("s1", datetime(2026, 1, 1), 0.7, 0.8, 1, 0.03).validate()
    with pytest.raises(ValueError):
        PaperSignal("s1", datetime(2026, 1, 1, tzinfo=timezone.utc), 1.2, 0.8, 1, 0.03).validate()
