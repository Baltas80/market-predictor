import numpy as np
import pandas as pd

from market_predictor.pipeline import run_final_lockbox_financial_comparison
from market_predictor.session_calendar import session_table


def make_frame(rows: int = 140) -> pd.DataFrame:
    index = session_table("2025-01-02", "2025-12-31")["close_utc"].iloc[:rows]
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0.2, 1.0, rows))
    close = np.maximum(close, 10)
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, rows)),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 10_000, rows),
            "macro_rate": rng.normal(0, 1, rows),
            "geo_pressure": rng.normal(0, 1, rows),
        },
        index=pd.DatetimeIndex(index),
    )


def test_financial_comparison_uses_same_lockbox_and_metrics():
    comparison, backtests = run_final_lockbox_financial_comparison(
        make_frame(),
        horizon=5,
        test_fraction=0.2,
        macro_features=["macro_rate"],
        geopolitical_features=["geo_pressure"],
        transaction_cost_bps=5,
        slippage_bps=1,
    )

    assert comparison["experiment"].tolist() == [
        "technical",
        "technical_macro",
        "technical_macro_geopolitical",
    ]
    assert len(comparison) == 3
    assert set(comparison.columns) >= {
        "total_return",
        "max_drawdown",
        "sharpe",
        "sortino",
        "turnover",
    }
    assert set(backtests) == set(comparison["experiment"])
    indices = [frame.index for frame in backtests.values()]
    assert all(index.equals(indices[0]) for index in indices[1:])
    assert len({len(frame) for frame in backtests.values()}) == 1


def test_financial_comparison_rejects_missing_external_feature():
    frame = make_frame().drop(columns="geo_pressure")
    try:
        run_final_lockbox_financial_comparison(
            frame,
            macro_features=["macro_rate"],
            geopolitical_features=["geo_pressure"],
        )
    except ValueError as exc:
        assert "geo_pressure" in str(exc)
    else:
        raise AssertionError("missing geopolitical feature should fail")
