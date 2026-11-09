import copy
import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional

from shot_probability import columns
from shot_probability.config import TransformerConfig
from shot_probability.geometry import PITCH_LENGTH, PITCH_WIDTH, FloatArray
from shot_probability.models import build_preprocessing

logger = logging.getLogger(__name__)

PADDING_ID = 0
UNKNOWN_ID = 1
FIRST_EVENT_TYPE_ID = 2
EVENT_NUMERIC_FEATURES = 4
SHOT_POSITION = 0


@dataclass(frozen=True)
class SequenceBatch:
    shot_features: Tensor
    event_type_ids: Tensor
    event_numeric: Tensor
    is_attacking: Tensor
    goals: Tensor

    def __len__(self) -> int:
        return len(self.goals)

    @property
    def is_event_valid(self) -> Tensor:
        return self.event_type_ids != PADDING_ID

    def select(self, indices: Tensor) -> "SequenceBatch":
        return SequenceBatch(
            shot_features=self.shot_features[indices],
            event_type_ids=self.event_type_ids[indices],
            event_numeric=self.event_numeric[indices],
            is_attacking=self.is_attacking[indices],
            goals=self.goals[indices],
        )


def build_event_type_ids(train: pd.DataFrame, sequence_length: int) -> dict[str, int]:
    """Vocabulary from the training split only, so validation and test cannot shape it."""
    type_columns = columns.sequence_columns(columns.SEQUENCE_EVENT_TYPE, sequence_length)
    event_types = pd.Series(train[type_columns].to_numpy().ravel()).dropna().unique()
    return {
        event_type: type_id
        for type_id, event_type in enumerate(sorted(event_types), start=FIRST_EVENT_TYPE_ID)
    }


def encode_sequences(
    shots: pd.DataFrame,
    shot_features: FloatArray,
    event_type_ids: dict[str, int],
    sequence_length: int,
) -> SequenceBatch:
    raw_types = shots[columns.sequence_columns(columns.SEQUENCE_EVENT_TYPE, sequence_length)]
    is_padding = raw_types.isna().to_numpy()
    mapped_ids = (
        pd.Series(raw_types.to_numpy().ravel())
        .map(event_type_ids)
        .fillna(UNKNOWN_ID)
        .to_numpy()
        .reshape(raw_types.shape)
    )
    type_ids = np.where(is_padding, PADDING_ID, mapped_ids)

    x_values = shots[columns.sequence_columns(columns.SEQUENCE_X, sequence_length)].to_numpy(float)
    y_values = shots[columns.sequence_columns(columns.SEQUENCE_Y, sequence_length)].to_numpy(float)
    seconds = shots[columns.sequence_columns(columns.SEQUENCE_SECONDS, sequence_length)].to_numpy(
        float
    )
    has_location = ~np.isnan(x_values) & ~np.isnan(y_values)
    event_numeric = np.stack(
        [
            np.nan_to_num(x_values / PITCH_LENGTH),
            np.nan_to_num(y_values / PITCH_WIDTH),
            has_location.astype(float),
            np.nan_to_num(np.log1p(seconds)),
        ],
        axis=-1,
    )
    is_attacking = shots[
        columns.sequence_columns(columns.SEQUENCE_IS_ATTACKING, sequence_length)
    ].to_numpy(int)
    return SequenceBatch(
        shot_features=torch.tensor(shot_features, dtype=torch.float32),
        event_type_ids=torch.tensor(type_ids, dtype=torch.long),
        event_numeric=torch.tensor(event_numeric, dtype=torch.float32),
        is_attacking=torch.tensor(is_attacking, dtype=torch.long),
        goals=torch.tensor(shots[columns.GOAL].to_numpy(float), dtype=torch.float32),
    )


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dimension: int, attention_heads: int, dropout: float) -> None:
        super().__init__()
        if model_dimension % attention_heads != 0:
            raise ValueError("model_dimension must be divisible by attention_heads")
        self.attention_heads = attention_heads
        self.head_dimension = model_dimension // attention_heads
        self.query_projection = nn.Linear(model_dimension, model_dimension)
        self.key_projection = nn.Linear(model_dimension, model_dimension)
        self.value_projection = nn.Linear(model_dimension, model_dimension)
        self.output_projection = nn.Linear(model_dimension, model_dimension)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tokens: Tensor) -> Tensor:
        batch_size, length, _ = tokens.shape
        return tokens.view(batch_size, length, self.attention_heads, self.head_dimension).transpose(
            1, 2
        )

    def attention_weights(self, tokens: Tensor, is_valid: Tensor) -> Tensor:
        """softmax(Q K^T / sqrt(d_head)) with padded keys masked out; shape (batch, heads, L, L)."""
        queries = self._split_heads(self.query_projection(tokens))
        keys = self._split_heads(self.key_projection(tokens))
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_dimension)
        scores = scores.masked_fill(~is_valid[:, None, None, :], float("-inf"))
        return torch.softmax(scores, dim=-1)

    def forward(self, tokens: Tensor, is_valid: Tensor) -> Tensor:
        weights = self.dropout(self.attention_weights(tokens, is_valid))
        values = self._split_heads(self.value_projection(tokens))
        batch_size, length, model_dimension = tokens.shape
        attended = (weights @ values).transpose(1, 2).reshape(batch_size, length, model_dimension)
        projected: Tensor = self.output_projection(attended)
        return projected


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.model_dimension)
        self.attention = MultiHeadSelfAttention(
            config.model_dimension, config.attention_heads, config.dropout
        )
        self.feedforward_norm = nn.LayerNorm(config.model_dimension)
        self.feedforward = nn.Sequential(
            nn.Linear(config.model_dimension, config.feedforward_dimension),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dimension, config.model_dimension),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, tokens: Tensor, is_valid: Tensor) -> Tensor:
        tokens = tokens + self.dropout(self.attention(self.attention_norm(tokens), is_valid))
        encoded: Tensor = tokens + self.dropout(self.feedforward(self.feedforward_norm(tokens)))
        return encoded


class ShotTransformer(nn.Module):
    """Encodes the shot as token 0 and its earlier possession events as tokens 1..K.

    The positional embedding is indexed by lag, so token `k` is the event `k` steps before the
    shot. The goal logit is read from the shot token only.
    """

    def __init__(
        self,
        config: TransformerConfig,
        shot_feature_count: int,
        event_type_count: int,
        initial_logit: float,
    ) -> None:
        super().__init__()
        dimension = config.model_dimension
        self.shot_projection = nn.Linear(shot_feature_count, dimension)
        self.event_type_embedding = nn.Embedding(
            event_type_count, dimension, padding_idx=PADDING_ID
        )
        self.team_side_embedding = nn.Embedding(2, dimension)
        self.event_numeric_projection = nn.Linear(EVENT_NUMERIC_FEATURES, dimension)
        self.lag_embedding = nn.Embedding(config.sequence_length + 1, dimension)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(EncoderLayer(config) for _ in range(config.encoder_layers))
        self.output_norm = nn.LayerNorm(dimension)
        self.goal_head = nn.Linear(dimension, 1)
        nn.init.constant_(self.goal_head.bias, initial_logit)

    def forward(self, batch: SequenceBatch) -> Tensor:
        shot_token = self.shot_projection(batch.shot_features).unsqueeze(1)
        event_tokens = (
            self.event_type_embedding(batch.event_type_ids)
            + self.team_side_embedding(batch.is_attacking)
            + self.event_numeric_projection(batch.event_numeric)
        )
        tokens = torch.cat([shot_token, event_tokens], dim=1)
        lags = torch.arange(tokens.shape[1], device=tokens.device)
        tokens = self.dropout(tokens + self.lag_embedding(lags))

        shot_is_valid = torch.ones(len(batch), 1, dtype=torch.bool, device=tokens.device)
        is_valid = torch.cat([shot_is_valid, batch.is_event_valid], dim=1)
        for layer in self.layers:
            tokens = layer(tokens, is_valid)
        shot_state = self.output_norm(tokens[:, SHOT_POSITION])
        logits: Tensor = self.goal_head(shot_state).squeeze(-1)
        return logits


def validation_log_loss(network: ShotTransformer, batch: SequenceBatch) -> float:
    network.eval()
    with torch.no_grad():
        return float(functional.binary_cross_entropy_with_logits(network(batch), batch.goals))


def fit_network(
    network: ShotTransformer,
    train_batch: SequenceBatch,
    validation_batch: SequenceBatch,
    config: TransformerConfig,
    seed: int,
) -> int:
    """Train with early stopping on validation log loss; returns the best epoch (1-based)."""
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    shuffler = torch.Generator().manual_seed(seed)
    best_loss = math.inf
    best_epoch = 0
    best_state = copy.deepcopy(network.state_dict())
    for epoch in range(1, config.max_epochs + 1):
        network.train()
        shuffled_indices = torch.randperm(len(train_batch), generator=shuffler)
        for start in range(0, len(train_batch), config.batch_size):
            mini_batch = train_batch.select(shuffled_indices[start : start + config.batch_size])
            loss = functional.binary_cross_entropy_with_logits(
                network(mini_batch), mini_batch.goals
            )
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        epoch_loss = validation_log_loss(network, validation_batch)
        if epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, epoch
            best_state = copy.deepcopy(network.state_dict())
        elif epoch - best_epoch >= config.patience:
            break
    network.load_state_dict(best_state)
    logger.info("Transformer best epoch=%d validation_log_loss=%.4f", best_epoch, best_loss)
    return best_epoch


class TransformerShotModel:
    def __init__(self, config: TransformerConfig, seed: int) -> None:
        self._config = config
        self._seed = seed
        self._shot_preprocessing = build_preprocessing(columns.SHOT_FEATURES)
        self._event_type_ids: dict[str, int] = {}
        self._network: ShotTransformer | None = None

    def _encode(self, shots: pd.DataFrame) -> SequenceBatch:
        shot_features = np.asarray(
            self._shot_preprocessing.transform(shots[columns.SHOT_FEATURES.names]), dtype=float
        )
        return encode_sequences(
            shots, shot_features, self._event_type_ids, self._config.sequence_length
        )

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> None:
        self._shot_preprocessing.fit(train[columns.SHOT_FEATURES.names])
        self._event_type_ids = build_event_type_ids(train, self._config.sequence_length)
        train_batch = self._encode(train)
        validation_batch = self._encode(validation)

        goal_rate = float(train[columns.GOAL].mean())
        torch.manual_seed(self._seed)
        network = ShotTransformer(
            self._config,
            shot_feature_count=train_batch.shot_features.shape[1],
            event_type_count=FIRST_EVENT_TYPE_ID + len(self._event_type_ids),
            initial_logit=math.log(goal_rate / (1.0 - goal_rate)),
        )
        fit_network(network, train_batch, validation_batch, self._config, self._seed)
        self._network = network

    def predict_goal_probability(self, shots: pd.DataFrame) -> FloatArray:
        if self._network is None:
            raise RuntimeError("fit must be called before predict_goal_probability")
        self._network.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(self._network(self._encode(shots)))
        return probabilities.numpy().astype(np.float64)
