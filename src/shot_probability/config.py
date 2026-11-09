from dataclasses import dataclass, field
from pathlib import Path

SHOTS_PATH = Path("data/shots.parquet")
CONTEXT_PATH = Path("data/shot_context.parquet")
FREEZE_FRAME_PATH = Path("data/shot_freeze_frame.parquet")
OUTPUTS_DIR = Path("outputs")

# Median shot follows 13 events in its possession; 16 covers 58% of shots completely and the
# uncapped scalar counts describe the rest. Chosen once, changing it requires `make data`.
SEQUENCE_LENGTH = 16


@dataclass(frozen=True)
class DataConfig:
    competition_id: int = 11
    season_id: int = 27
    fetch_workers: int = 8
    sequence_length: int = SEQUENCE_LENGTH
    shots_path: Path = SHOTS_PATH
    context_path: Path = CONTEXT_PATH
    freeze_frame_path: Path = FREEZE_FRAME_PATH


@dataclass(frozen=True)
class GradientBoostingConfig:
    learning_rate: float = 0.05
    max_depth: int = 3
    min_samples_leaf: int = 50
    l2_regularization: float = 1.0
    max_iterations: int = 400


@dataclass(frozen=True)
class TransformerConfig:
    sequence_length: int = SEQUENCE_LENGTH
    model_dimension: int = 32
    attention_heads: int = 4
    encoder_layers: int = 2
    feedforward_dimension: int = 64
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 128
    max_epochs: int = 100
    patience: int = 10


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 42
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    regularization_strength: float = 1.0
    max_iter: int = 1000
    calibration_bins: int = 10
    bootstrap_resamples: int = 2000
    confidence_level: float = 0.95
    gradient_boosting: GradientBoostingConfig = field(default_factory=GradientBoostingConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    shots_path: Path = SHOTS_PATH
    context_path: Path = CONTEXT_PATH
    freeze_frame_path: Path = FREEZE_FRAME_PATH
    outputs_dir: Path = OUTPUTS_DIR
