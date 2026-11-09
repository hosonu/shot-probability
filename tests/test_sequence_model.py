import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import log_loss

from shot_probability import columns
from shot_probability.config import TransformerConfig
from shot_probability.models import PriorModel
from shot_probability.sequence_model import (
    PADDING_ID,
    UNKNOWN_ID,
    MultiHeadSelfAttention,
    SequenceBatch,
    ShotTransformer,
    TransformerShotModel,
    build_event_type_ids,
    encode_sequences,
)
from synthetic import TEST_SEQUENCE_LENGTH, make_benchmark_table

SEED = 3
TEST_CONFIG = TransformerConfig(
    sequence_length=TEST_SEQUENCE_LENGTH,
    model_dimension=16,
    attention_heads=2,
    encoder_layers=1,
    feedforward_dimension=32,
    dropout=0.0,
    batch_size=64,
    max_epochs=60,
    patience=10,
)


def make_batch(shot_count: int = 6, shot_feature_count: int = 5) -> SequenceBatch:
    generator = torch.Generator().manual_seed(0)
    event_type_ids = torch.randint(2, 5, (shot_count, TEST_SEQUENCE_LENGTH), generator=generator)
    event_type_ids[:, -2:] = PADDING_ID
    return SequenceBatch(
        shot_features=torch.randn(shot_count, shot_feature_count, generator=generator),
        event_type_ids=event_type_ids,
        event_numeric=torch.randn(shot_count, TEST_SEQUENCE_LENGTH, 4, generator=generator),
        is_attacking=torch.randint(0, 2, (shot_count, TEST_SEQUENCE_LENGTH), generator=generator),
        goals=torch.zeros(shot_count),
    )


def test_attention_weights_are_a_softmax_that_ignores_padded_keys() -> None:
    attention = MultiHeadSelfAttention(model_dimension=16, attention_heads=2, dropout=0.0)
    tokens = torch.randn(3, 5, 16)
    is_valid = torch.tensor(
        [[True] * 5, [True, True, True, False, False], [True] * 3 + [False] * 2]
    )

    weights = attention.attention_weights(tokens, is_valid)

    assert weights.shape == (3, 2, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 2, 5))
    assert (weights[1, :, :, 3:] == 0).all()


def test_attention_rejects_a_model_dimension_that_heads_cannot_split() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadSelfAttention(model_dimension=10, attention_heads=4, dropout=0.0)


def test_prediction_does_not_depend_on_the_content_of_padded_slots() -> None:
    torch.manual_seed(0)
    network = ShotTransformer(
        TEST_CONFIG, shot_feature_count=5, event_type_count=6, initial_logit=-2.0
    )
    network.eval()
    batch = make_batch()
    scrambled = dataclasses.replace(
        batch,
        event_numeric=torch.where(
            batch.is_event_valid.unsqueeze(-1), batch.event_numeric, batch.event_numeric * 100 + 7
        ),
        is_attacking=torch.where(batch.is_event_valid, batch.is_attacking, 1 - batch.is_attacking),
    )

    with torch.no_grad():
        assert torch.allclose(network(batch), network(scrambled), atol=1e-6)


def test_encoding_pads_unknown_types_and_missing_locations() -> None:
    shots = make_benchmark_table(shot_count=40)
    event_type_ids = build_event_type_ids(shots, TEST_SEQUENCE_LENGTH)
    unseen_shots = shots.copy()
    type_column = columns.sequence_column(columns.SEQUENCE_EVENT_TYPE, 1)
    unseen_shots[type_column] = "Dribble"

    batch = encode_sequences(
        unseen_shots,
        np.zeros((len(shots), 5)),
        event_type_ids,
        TEST_SEQUENCE_LENGTH,
    )

    assert (batch.event_type_ids[:, 0] == UNKNOWN_ID).all()
    assert set(event_type_ids) == {"Pass", "Carry", "Pressure"}
    assert min(event_type_ids.values()) > UNKNOWN_ID
    padded_lag_four = shots[columns.sequence_column(columns.SEQUENCE_EVENT_TYPE, 4)].isna()
    assert (
        batch.event_type_ids[:, 3][torch.tensor(padded_lag_four.to_numpy())] == PADDING_ID
    ).all()
    assert (batch.event_numeric[~batch.is_event_valid] == 0).all()


def test_transformer_learns_a_signal_that_only_exists_in_the_sequence() -> None:
    shots = make_benchmark_table(shot_count=1500, signal_source="sequence")
    train, validation, test = shots.iloc[:900], shots.iloc[900:1200], shots.iloc[1200:]
    model = TransformerShotModel(TEST_CONFIG, SEED)
    prior = PriorModel()

    model.fit(train, validation)
    prior.fit(train, validation)

    transformer_loss = log_loss(test[columns.GOAL], model.predict_goal_probability(test))
    prior_loss = log_loss(test[columns.GOAL], prior.predict_goal_probability(test))
    assert transformer_loss < 0.5 * prior_loss


def test_same_seed_gives_identical_predictions() -> None:
    shots = make_benchmark_table(shot_count=300)
    train, validation = shots.iloc[:200], shots.iloc[200:]
    config = dataclasses.replace(TEST_CONFIG, max_epochs=3)
    first, second = TransformerShotModel(config, SEED), TransformerShotModel(config, SEED)

    first.fit(train, validation)
    second.fit(train, validation)

    np.testing.assert_array_equal(
        first.predict_goal_probability(validation), second.predict_goal_probability(validation)
    )


def test_predicting_before_fitting_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        TransformerShotModel(TEST_CONFIG, SEED).predict_goal_probability(pd.DataFrame())
