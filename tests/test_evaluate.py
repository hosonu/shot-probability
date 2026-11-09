import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from shot_probability.evaluate import bootstrap_log_loss_difference, per_shot_log_loss


def make_predictions() -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    match_ids = pd.Series(np.repeat(np.arange(60), 10))
    true_probability = rng.uniform(0.02, 0.6, len(match_ids))
    goals = pd.Series((rng.uniform(size=len(match_ids)) < true_probability).astype(int))
    return goals, match_ids, true_probability, np.full(len(match_ids), goals.mean())


def test_per_shot_log_loss_matches_sklearn() -> None:
    goals, _, probabilities, _ = make_predictions()

    assert per_shot_log_loss(goals.to_numpy(float), probabilities).mean() == pytest.approx(
        log_loss(goals, probabilities)
    )


def test_identical_models_have_a_zero_difference_and_a_zero_width_interval() -> None:
    goals, match_ids, probabilities, _ = make_predictions()

    difference = bootstrap_log_loss_difference(
        goals, match_ids, probabilities, probabilities, 100, 0.95, seed=0
    )

    assert (difference.mean, difference.lower, difference.upper) == (0.0, 0.0, 0.0)


def test_a_better_candidate_has_a_negative_difference_and_an_interval_below_zero() -> None:
    goals, match_ids, informative, uninformative = make_predictions()

    difference = bootstrap_log_loss_difference(
        goals, match_ids, uninformative, informative, 500, 0.95, seed=0
    )

    assert difference.upper < 0.0
    assert difference.lower <= difference.mean <= difference.upper


def test_bootstrap_is_reproducible_for_a_fixed_seed() -> None:
    goals, match_ids, informative, uninformative = make_predictions()

    first = bootstrap_log_loss_difference(goals, match_ids, uninformative, informative, 200, 0.9, 5)
    second = bootstrap_log_loss_difference(
        goals, match_ids, uninformative, informative, 200, 0.9, 5
    )

    assert first == second
