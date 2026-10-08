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
