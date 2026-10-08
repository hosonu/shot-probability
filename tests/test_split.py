import numpy as np
import pandas as pd

from shot_probability import columns
from shot_probability.split import split_by_match


def make_shots(match_count: int = 40, shots_per_match: int = 5) -> pd.DataFrame:
    return pd.DataFrame({columns.MATCH_ID: np.repeat(np.arange(match_count), shots_per_match)})


def test_no_match_appears_in_more_than_one_partition() -> None:
    split = split_by_match(make_shots(), validation_fraction=0.15, test_fraction=0.15, seed=0)

    train_ids = set(split.train[columns.MATCH_ID])
    validation_ids = set(split.validation[columns.MATCH_ID])
    test_ids = set(split.test[columns.MATCH_ID])
    assert not train_ids & validation_ids
    assert not train_ids & test_ids
    assert not validation_ids & test_ids


def test_every_shot_is_assigned_exactly_once() -> None:
    shots = make_shots()
    split = split_by_match(shots, validation_fraction=0.15, test_fraction=0.15, seed=0)

    assert len(split.train) + len(split.validation) + len(split.test) == len(shots)


def test_split_is_reproducible_for_a_fixed_seed() -> None:
    first = split_by_match(make_shots(), validation_fraction=0.2, test_fraction=0.2, seed=3)
    second = split_by_match(make_shots(), validation_fraction=0.2, test_fraction=0.2, seed=3)

    pd.testing.assert_frame_equal(first.test, second.test)
