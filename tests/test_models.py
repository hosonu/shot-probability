import numpy as np
import pandas as pd

from shot_probability import columns
from shot_probability.benchmark import build_models
from shot_probability.models import (
    PriorModel,
    build_gradient_boosting_model,
    build_logistic_model,
)
from synthetic import make_benchmark_table, make_model_config


def split_table(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return table.iloc[:400], table.iloc[400:500], table.iloc[500:]


def test_prior_model_predicts_the_training_goal_rate() -> None:
    train, validation, test = split_table(make_benchmark_table())
    model = PriorModel()

    model.fit(train, validation)

    np.testing.assert_allclose(model.predict_goal_probability(test), train[columns.GOAL].mean())


def test_gradient_boosting_picks_its_rounds_on_validation_and_stays_within_the_cap() -> None:
    config = make_model_config()
    train, validation, test = split_table(make_benchmark_table())
    model = build_gradient_boosting_model(config, columns.SHOT_FEATURES)

    model.fit(train, validation)

    assert 1 <= model.iteration_count <= config.gradient_boosting.max_iterations
    probabilities = model.predict_goal_probability(test)
    assert len(probabilities) == len(test)
    assert ((probabilities > 0) & (probabilities < 1)).all()


def test_logistic_baseline_ignores_every_context_column() -> None:
    config = make_model_config()
    train, validation, test = split_table(make_benchmark_table())
    scrambled_test = test.assign(
        **{
            column: test[column].sample(frac=1.0, random_state=1).to_numpy()
            for column in columns.CONTEXT_SCALAR_COLUMNS
        }
    )
    model = build_logistic_model(config, columns.SHOT_FEATURES)
    model.fit(train, validation)

    np.testing.assert_array_equal(
        model.predict_goal_probability(test), model.predict_goal_probability(scrambled_test)
    )


def test_context_models_tolerate_an_event_type_unseen_in_training() -> None:
    config = make_model_config()
    train, validation, test = split_table(make_benchmark_table())
    model = build_logistic_model(config, columns.CONTEXT_FEATURES)
    model.fit(train, validation)

    probabilities = model.predict_goal_probability(
        test.assign(**{columns.LAST_EVENT_TYPE: "Dribble"})
    )

    assert np.isfinite(probabilities).all()


def test_every_benchmarked_model_returns_one_probability_per_shot() -> None:
    config = make_model_config()
    train, validation, test = split_table(make_benchmark_table())

    for model in build_models(config).values():
        model.fit(train, validation)
        probabilities = model.predict_goal_probability(test)
        assert probabilities.shape == (len(test),)
        assert ((probabilities > 0) & (probabilities < 1)).all()


def test_no_model_input_can_expose_the_outcome_or_statsbomb_xg() -> None:
    forbidden = {columns.GOAL, "shot_outcome", "statsbomb_xg", columns.EVENT_ID, columns.MATCH_ID}

    assert not forbidden & set(columns.CONTEXT_FEATURES.names)
