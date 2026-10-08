from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from shot_probability import columns

FloatArray = NDArray[np.float64]
DECISION_THRESHOLD = 0.5


def score_predictions(
    model_name: str, split_name: str, goals: pd.Series, goal_probabilities: FloatArray
) -> dict[str, str | int | float]:
    return {
        "model": model_name,
        "split": split_name,
        "n_shots": len(goals),
        "goal_rate": float(goals.mean()),
        "log_loss": float(log_loss(goals, goal_probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(goals, goal_probabilities)),
        "accuracy": float(accuracy_score(goals, goal_probabilities >= DECISION_THRESHOLD)),
    }


def save_calibration_plot(
    shots_by_split: dict[str, pd.DataFrame],
    probabilities_by_split: dict[str, FloatArray],
    n_bins: int,
    path: Path,
) -> None:
    figure = Figure(figsize=(5 * len(shots_by_split), 4.5))
    axes = figure.subplots(1, len(shots_by_split), sharex=True, sharey=True, squeeze=False)[0]
    for axis, (split_name, split_shots) in zip(axes, shots_by_split.items(), strict=True):
        observed_rate, predicted_rate = calibration_curve(
            split_shots[columns.GOAL],
            probabilities_by_split[split_name],
            n_bins=n_bins,
            strategy="quantile",
        )
        axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfectly calibrated")
        axis.plot(predicted_rate, observed_rate, marker="o", label="logistic")
        axis.set_title(f"{split_name} (n={len(split_shots)})")
        axis.set_xlabel("mean predicted goal probability")
        axis.set_ylabel("observed goal rate")
        axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
