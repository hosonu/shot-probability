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


MULTI_LEAGUE_DIR = Path("data/multi_league")
MULTI_LEAGUE_OUTPUTS_DIR = Path("outputs/multi_league")


@dataclass(frozen=True)
class LeagueConfig:
    """A competition named as StatsBomb's ``sb.competitions()`` lists it; ids are looked up."""

    competition_name: str
    country_name: str


@dataclass(frozen=True)
class MultiLeagueDataConfig:
    season_name: str = "2015/2016"
    leagues: tuple[LeagueConfig, ...] = (
        LeagueConfig("La Liga", "Spain"),
        LeagueConfig("Premier League", "England"),
        LeagueConfig("Serie A", "Italy"),
        LeagueConfig("Ligue 1", "France"),
    )
    # The published single-league benchmark; its rows must be reproduced exactly.
    legacy_league: str = "La Liga"
    fetch_workers: int = 8
    sequence_length: int = SEQUENCE_LENGTH
    shots_path: Path = MULTI_LEAGUE_DIR / "shots.parquet"
    context_path: Path = MULTI_LEAGUE_DIR / "shot_context.parquet"
    freeze_frame_path: Path = MULTI_LEAGUE_DIR / "shot_freeze_frame.parquet"
    attributes_path: Path = MULTI_LEAGUE_DIR / "shot_attributes.parquet"
    legacy_shots_path: Path = SHOTS_PATH
    legacy_context_path: Path = CONTEXT_PATH
    legacy_freeze_frame_path: Path = FREEZE_FRAME_PATH


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


@dataclass(frozen=True)
class MultiLeagueExperimentConfig:
    """Settings shared by the learning curve and the information-group comparison.

    Model hyperparameters come from ``model`` unchanged; nothing is searched on validation.
    """

    model: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    data: MultiLeagueDataConfig = field(default_factory=MultiLeagueDataConfig)
    # Learning curve: one match-level split pooled over all leagues, nested training subsets.
    learning_curve_train_shots: tuple[int, ...] = (4500, 9000, 13500, 18000)
    # Fixed by the learning curve (see README); the other model is kept as a robustness check.
    primary_model: str = "logistic"
    check_model: str = "gradient_boosting"
    outputs_dir: Path = MULTI_LEAGUE_OUTPUTS_DIR
