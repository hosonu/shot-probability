import dataclasses
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from shot_probability import columns
from shot_probability.config import BenchmarkConfig, GradientBoostingConfig, TransformerConfig

TEST_SEQUENCE_LENGTH = 4
EVENT_TYPES = np.array(["Pass", "Carry", "Pressure"], dtype=object)

SignalSource = Literal["location", "sequence"]


def make_model_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        bootstrap_resamples=50,
        calibration_bins=3,
        gradient_boosting=GradientBoostingConfig(max_iterations=20),
        transformer=TransformerConfig(
            sequence_length=TEST_SEQUENCE_LENGTH,
            model_dimension=16,
            attention_heads=2,
            encoder_layers=1,
            feedforward_dimension=32,
            batch_size=64,
            max_epochs=3,
            patience=2,
        ),
    )


def make_test_config(tmp_path: Path) -> BenchmarkConfig:
    return dataclasses.replace(
        make_model_config(),
        shots_path=tmp_path / "shots.parquet",
        context_path=tmp_path / "shot_context.parquet",
        freeze_frame_path=tmp_path / "shot_freeze_frame.parquet",
        outputs_dir=tmp_path / "outputs",
    )


def make_benchmark_table(
    shot_count: int = 600, signal_source: SignalSource = "location", seed: int = 0
) -> pd.DataFrame:
    """Synthetic shots whose goal depends either on distance or on the event two lags back."""
    rng = np.random.default_rng(seed)
    distance = rng.uniform(5.0, 35.0, shot_count)
    prior_event_count = rng.integers(0, TEST_SEQUENCE_LENGTH + 3, shot_count)
    lags = np.arange(1, TEST_SEQUENCE_LENGTH + 1)
    is_available = lags[None, :] <= prior_event_count[:, None]
    event_types = np.where(
        is_available,
        rng.choice(EVENT_TYPES, size=(shot_count, TEST_SEQUENCE_LENGTH)),
        np.full(is_available.shape, None, dtype=object),
    )
    if signal_source == "location":
        goal_probability = 1.0 / (1.0 + np.exp(0.3 * (distance - 15.0)))
        goals = (rng.uniform(size=shot_count) < goal_probability).astype(int)
    else:
        goals = (event_types[:, 1] == "Carry").astype(int)

    table = pd.DataFrame(
        {
            columns.MATCH_ID: np.arange(shot_count) // 5,
            columns.EVENT_ID: [f"event-{index}" for index in range(shot_count)],
            columns.DISTANCE_TO_GOAL: distance,
            columns.ANGLE_TO_GOAL: 1.0 / distance,
            columns.BODY_PART: rng.choice(["foot", "head", "other"], shot_count),
            columns.IS_SET_PIECE: rng.uniform(size=shot_count) < 0.1,
            columns.GOAL: goals,
            columns.N_PRIOR_EVENTS: prior_event_count,
            columns.N_PRIOR_PASSES: rng.integers(0, 5, shot_count),
            columns.N_PRIOR_CARRIES: rng.integers(0, 5, shot_count),
            columns.N_PRIOR_DEFENDING_EVENTS: rng.integers(0, 5, shot_count),
            columns.POSSESSION_SECONDS_BEFORE_SHOT: rng.uniform(0.0, 30.0, shot_count),
            columns.SECONDS_SINCE_PREVIOUS_EVENT: rng.uniform(0.0, 3.0, shot_count),
            columns.PROGRESSION_IN_WINDOW: rng.uniform(-10.0, 40.0, shot_count),
            columns.LAST_EVENT_TYPE: np.where(is_available[:, 0], event_types[:, 0], "none"),
            columns.HAS_FREEZE_FRAME: True,
            columns.HAS_GOALKEEPER: True,
            columns.GOALKEEPER_DISTANCE_TO_GOAL: rng.uniform(0.5, 8.0, shot_count),
            columns.GOALKEEPER_Y_OFFSET: rng.uniform(0.0, 5.0, shot_count),
            columns.SHOT_TO_GOALKEEPER_DISTANCE: rng.uniform(2.0, 30.0, shot_count),
            columns.NEAREST_OPPONENT_DISTANCE: rng.uniform(0.5, 20.0, shot_count),
            columns.OPPONENTS_IN_SHOT_TRIANGLE: rng.integers(0, 4, shot_count),
            columns.TEAMMATES_IN_SHOT_TRIANGLE: rng.integers(0, 3, shot_count),
            columns.OPPONENTS_BETWEEN_SHOT_AND_GOAL: rng.integers(0, 6, shot_count),
        }
    )
    sequence_values = {
        columns.SEQUENCE_EVENT_TYPE: event_types,
        columns.SEQUENCE_X: np.where(
            is_available, rng.uniform(0.0, 120.0, event_types.shape), np.nan
        ),
        columns.SEQUENCE_Y: np.where(
            is_available, rng.uniform(0.0, 80.0, event_types.shape), np.nan
        ),
        columns.SEQUENCE_SECONDS: np.where(
            is_available, rng.uniform(0.0, 3.0, event_types.shape), np.nan
        ),
        columns.SEQUENCE_IS_ATTACKING: is_available & (rng.uniform(size=event_types.shape) < 0.7),
    }
    sequence_frames = [
        pd.DataFrame(values, columns=columns.sequence_columns(field, TEST_SEQUENCE_LENGTH))
        for field, values in sequence_values.items()
    ]
    return pd.concat([table, *sequence_frames], axis=1)


def write_benchmark_table(table: pd.DataFrame, config: BenchmarkConfig) -> None:
    key_columns = [columns.MATCH_ID, columns.EVENT_ID]
    freeze_frame_columns = key_columns + columns.FREEZE_FRAME_SCALAR_COLUMNS
    context_columns = [
        column
        for column in table.columns
        if column in key_columns
        or (
            column not in columns.SHOT_TABLE_COLUMNS
            and column not in columns.FREEZE_FRAME_SCALAR_COLUMNS
        )
    ]
    table[columns.SHOT_TABLE_COLUMNS].to_parquet(config.shots_path, index=False)
    table[context_columns].to_parquet(config.context_path, index=False)
    table[freeze_frame_columns].to_parquet(config.freeze_frame_path, index=False)
