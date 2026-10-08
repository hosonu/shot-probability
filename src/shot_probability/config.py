from dataclasses import dataclass
from pathlib import Path

SHOTS_PATH = Path("data/shots.parquet")
OUTPUTS_DIR = Path("outputs")


@dataclass(frozen=True)
class DataConfig:
    competition_id: int = 11
    season_id: int = 27
    fetch_workers: int = 8
    shots_path: Path = SHOTS_PATH


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    regularization_strength: float = 1.0
    max_iter: int = 1000
    calibration_bins: int = 10
    shots_path: Path = SHOTS_PATH
    outputs_dir: Path = OUTPUTS_DIR
