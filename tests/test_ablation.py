import pandas as pd
import pytest

from market_predictor.ablation import build_ablation_specs, run_common_fold_ablation


def test_ablation_specs_are_nested_a_b_c():
    specs = build_ablation_specs(technical=("r1",), macro=("gdp",), events=("event_total_pressure",))
    assert [s.name for s in specs] == ["A", "B", "C"]
    assert specs[1].feature_columns == ("r1", "gdp")
    assert specs[2].feature_columns == ("r1", "gdp", "event_total_pressure")


def test_ablation_uses_same_data_and_evaluator_contract():
    data = pd.DataFrame({"r1":[1], "gdp":[2], "event_total_pressure":[3]})
    calls = []
    def evaluator(frame, features):
        calls.append((id(frame), features))
        return {"accuracy": len(features) / 3}
    result = run_common_fold_ablation(
        data,
        build_ablation_specs(technical=("r1",), macro=("gdp",), events=("event_total_pressure",)),
        evaluator,
    )
    assert len(result) == 3
    assert len({call[0] for call in calls}) == 1


def test_ablation_rejects_missing_features():
    with pytest.raises(ValueError):
        run_common_fold_ablation(
            pd.DataFrame({"r1":[1]}),
            build_ablation_specs(technical=("r1",), macro=("gdp",), events=()),
            lambda *_: {"accuracy": 0.5},
        )
