from dataclasses import dataclass

import numpy as np
import pandas as pd

from shot_probability import columns


@dataclass(frozen=True)
class ShotSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_by_match(
    shots: pd.DataFrame, validation_fraction: float, test_fraction: float, seed: int
) -> ShotSplit:
    """Partition shots so every match lands in exactly one of train, validation, test."""
    match_ids = np.sort(shots[columns.MATCH_ID].unique())
    shuffled_match_ids = np.random.default_rng(seed).permutation(match_ids)

    test_count = round(len(match_ids) * test_fraction)
    validation_count = round(len(match_ids) * validation_fraction)
    test_ids = shuffled_match_ids[:test_count]
    validation_ids = shuffled_match_ids[test_count : test_count + validation_count]

    is_test = shots[columns.MATCH_ID].isin(test_ids)
    is_validation = shots[columns.MATCH_ID].isin(validation_ids)
    return ShotSplit(
        train=shots.loc[~is_test & ~is_validation],
        validation=shots.loc[is_validation],
        test=shots.loc[is_test],
    )


def split_leave_one_league_out(
    shots: pd.DataFrame, test_league: str, validation_fraction: float, seed: int
) -> ShotSplit:
    """Test on one whole league; validation is a match-level slice of the training leagues."""
    is_test = shots[columns.LEAGUE] == test_league
    if not is_test.any():
        raise ValueError(f"No shots for test league {test_league!r}")
    training_leagues = split_by_match(
        shots.loc[~is_test], validation_fraction, test_fraction=0.0, seed=seed
    )
    return ShotSplit(
        train=training_leagues.train,
        validation=training_leagues.validation,
        test=shots.loc[is_test],
    )


def subsample_matches(shots: pd.DataFrame, target_shot_count: int, seed: int) -> pd.DataFrame:
    """Smallest prefix of a seeded match permutation holding at least ``target_shot_count`` shots.

    The permutation depends only on ``seed`` and the match ids, so larger targets are supersets
    of smaller ones (nested training sets for a learning curve). Whole matches are kept.
    """
    match_ids = np.sort(shots[columns.MATCH_ID].unique())
    shuffled_match_ids = np.random.default_rng(seed).permutation(match_ids)
    shots_per_match = shots[columns.MATCH_ID].value_counts().reindex(shuffled_match_ids)
    cumulative_shots = shots_per_match.cumsum().to_numpy()
    match_count = int(np.searchsorted(cumulative_shots, target_shot_count)) + 1
    return shots.loc[shots[columns.MATCH_ID].isin(shuffled_match_ids[:match_count])]
