import dataclasses

import numpy as np
import pandas as pd
import pytest

from shot_probability import columns
from shot_probability.config import MultiLeagueExperimentConfig
from shot_probability.multi_league import (
    ALL_SET,
    BASE_SET,
    GRADIENT_BOOSTING,
    LOGISTIC,
    POOLED_SCOPE,
    STATSBOMB_REFERENCE,
    build_comparisons,
    build_group_feature_sets,
    choose_primary_model,
    run_group_comparison,
    run_learning_curve,
)
from shot_probability.split import split_leave_one_league_out, subsample_matches
from synthetic import make_benchmark_table, make_model_config

LEAGUE_NAMES = ["A", "B", "C"]


def make_multi_league_table(shot_count: int = 900) -> pd.DataFrame:
    table = make_benchmark_table(shot_count=shot_count)
    rng = np.random.default_rng(1)
    match_count = table[columns.MATCH_ID].max() + 1
    league_of_match = np.array(LEAGUE_NAMES)[np.arange(match_count) % len(LEAGUE_NAMES)]
    return table.assign(
        **{
            columns.LEAGUE: league_of_match[table[columns.MATCH_ID].to_numpy()],
            columns.IS_PENALTY: rng.uniform(size=shot_count) < 0.02,
            columns.SHOT_TECHNIQUE: rng.choice(["Normal", "Volley"], shot_count),
            columns.IS_FIRST_TIME: rng.uniform(size=shot_count) < 0.3,
            columns.IS_AERIAL_WON: rng.uniform(size=shot_count) < 0.1,
            columns.PLAY_PATTERN: rng.choice(["Regular Play", "From Corner"], shot_count),
            columns.IS_UNDER_PRESSURE: rng.uniform(size=shot_count) < 0.4,
            columns.STATSBOMB_XG: rng.uniform(0.01, 0.6, shot_count),
        }
    )


def make_experiment_config() -> MultiLeagueExperimentConfig:
    return MultiLeagueExperimentConfig(
        model=make_model_config(), learning_curve_train_shots=(150, 300)
    )


def test_leave_one_league_out_tests_on_exactly_one_league() -> None:
    table = make_multi_league_table()

    split = split_leave_one_league_out(table, "B", validation_fraction=0.2, seed=0)

    assert set(split.test[columns.LEAGUE]) == {"B"}
    assert "B" not in set(split.train[columns.LEAGUE]) | set(split.validation[columns.LEAGUE])
    assert not set(split.train[columns.MATCH_ID]) & set(split.validation[columns.MATCH_ID])
    assert len(split.train) + len(split.validation) + len(split.test) == len(table)


def test_unknown_test_league_fails() -> None:
    with pytest.raises(ValueError, match="No shots"):
        split_leave_one_league_out(make_multi_league_table(), "Z", 0.2, seed=0)


def test_subsamples_are_nested_whole_matches() -> None:
    table = make_multi_league_table()

    small = subsample_matches(table, 100, seed=3)
    large = subsample_matches(table, 400, seed=3)

    assert len(small) >= 100
    assert len(large) >= 400
    assert set(small[columns.EVENT_ID]) <= set(large[columns.EVENT_ID])
    small_matches = table[columns.MATCH_ID].isin(small[columns.MATCH_ID].unique())
    assert small_matches.sum() == len(small)


def test_group_feature_sets_add_and_remove_one_group_at_a_time() -> None:
    feature_sets = build_group_feature_sets(columns.FEATURE_GROUPS)

    assert feature_sets[BASE_SET] == columns.SHOT_FEATURES
    assert set(feature_sets["+penalty"].names) == {*columns.SHOT_FEATURES.names, columns.IS_PENALTY}
    assert columns.IS_PENALTY not in feature_sets["-penalty"].names
    assert set(feature_sets["-penalty"].names) | {columns.IS_PENALTY} == set(
        feature_sets[ALL_SET].names
    )
    for feature_set in feature_sets.values():
        assert columns.STATSBOMB_XG not in feature_set.names
        assert len(feature_set.names) == len(set(feature_set.names))


def test_comparisons_include_context_on_top_of_freeze_frame() -> None:
    comparisons = build_comparisons(columns.FEATURE_GROUPS)

    combination = [comparison for comparison in comparisons if comparison.kind == "combination"]
    assert len(combination) == 1
    assert combination[0].candidate == "+context+freeze_frame"
    assert combination[0].baseline == "+freeze_frame"
    assert {comparison.kind for comparison in comparisons} == {"add", "remove", "combination"}


def test_primary_model_is_boosting_only_when_it_wins_clearly_at_the_largest_size() -> None:
    curve = pd.DataFrame(
        {
            "feature_set": ["shot", "shot", "full", "full"],
            "train_shots": [100, 200, 100, 200],
            "ci_upper": [0.01, -0.001, 0.02, 0.001],
        }
    )
    assert choose_primary_model(curve) == LOGISTIC
    assert choose_primary_model(curve.assign(ci_upper=-0.001)) == GRADIENT_BOOSTING


def test_learning_curve_grows_training_set_with_fixed_test() -> None:
    curve = run_learning_curve(make_multi_league_table(), make_experiment_config())

    for _, rows in curve.groupby("feature_set"):
        assert rows["train_shots"].is_monotonic_increasing
        assert rows["train_shots"].nunique() == len(rows)
    assert (curve["ci_lower"] <= curve["ci_upper"]).all()


def test_group_comparison_scores_every_shot_out_of_league() -> None:
    table = make_multi_league_table()
    groups = {name: columns.FEATURE_GROUPS[name] for name in ("freeze_frame", "penalty")}

    metrics, comparison = run_group_comparison(table, groups, make_experiment_config())

    assert set(metrics["test_league"]) == {POOLED_SCOPE, *LEAGUE_NAMES}
    assert STATSBOMB_REFERENCE in set(metrics["model"])
    assert set(comparison["model"]) == {LOGISTIC, GRADIENT_BOOSTING}
    assert set(comparison["candidate"]) == {
        "+freeze_frame",
        "+penalty",
        "-freeze_frame",
        "-penalty",
    }


def test_group_comparison_does_not_depend_on_statsbomb_xg() -> None:
    table = make_multi_league_table()
    groups = {"penalty": columns.FEATURE_GROUPS["penalty"]}
    config = dataclasses.replace(make_experiment_config(), check_model=LOGISTIC)

    _, original = run_group_comparison(table, groups, config)
    shuffled_xg = table[columns.STATSBOMB_XG].sample(frac=1.0, random_state=0).to_numpy()
    _, altered = run_group_comparison(
        table.assign(**{columns.STATSBOMB_XG: shuffled_xg}), groups, config
    )

    pd.testing.assert_frame_equal(original, altered)
