import numpy as np
import pandas as pd

from market_predictor.event_schema import EventCategory, MarketEvent
from market_predictor.pipeline import run_final_lockbox_event_experiments


def _market_data(n: int = 160) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.sin(np.arange(n) / 7.0) + 0.2)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=index,
    )


def test_event_pipeline_adds_events_only_to_c_and_keeps_same_lockbox():
    df = _market_data()
    events = [
        MarketEvent(
            "c1",
            EventCategory.CORRUPTION,
            "2020-04-15T12:00:00Z",
            0.9,
            surprise=0.8,
        )
    ]
    results = run_final_lockbox_event_experiments(
        df,
        events,
        event_features=["event_corruption", "event_total_pressure", "event_surprise"],
    )

    assert [result.name for result in results] == [
        "technical",
        "technical_macro",
        "technical_macro_geopolitical",
    ]
    assert all(len(result.predictions) == len(results[0].predictions) for result in results)
    assert all(result.predictions.index.equals(results[0].predictions.index) for result in results)


def test_event_pipeline_rejects_non_datetime_market_index():
    df = _market_data().reset_index(drop=True)
    try:
        run_final_lockbox_event_experiments(df, [])
    except TypeError as exc:
        assert "DatetimeIndex" in str(exc)
    else:
        raise AssertionError("expected TypeError")
