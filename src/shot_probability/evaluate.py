from dataclasses import dataclass
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
PROBABILITY_EPSILON = 1e-15


@dataclass(frozen=True)
class LogLossDifference:
    """Candidate minus baseline log loss. Negative means the candidate is better."""

    mean: float
    lower: float
    upper: float


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


def per_shot_log_loss(goals: FloatArray, goal_probabilities: FloatArray) -> FloatArray:
    clipped = np.clip(goal_probabilities, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return -(goals * np.log(clipped) + (1.0 - goals) * np.log(1.0 - clipped))


def bootstrap_log_loss_difference(
    goals: pd.Series,
    match_ids: pd.Series,
    baseline_probabilities: FloatArray,
    candidate_probabilities: FloatArray,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> LogLossDifference:
    """Paired bootstrap that resamples whole matches, because shots within a match are related."""
    goal_values = goals.to_numpy(dtype=float)
    shot_differences = per_shot_log_loss(goal_values, candidate_probabilities) - per_shot_log_loss(
        goal_values, baseline_probabilities
    )
    _, match_codes = np.unique(match_ids.to_numpy(), return_inverse=True)
    difference_per_match = np.bincount(match_codes, weights=shot_differences)
    shots_per_match = np.bincount(match_codes)

    sampled_matches = np.random.default_rng(seed).integers(
        0, len(shots_per_match), size=(n_resamples, len(shots_per_match))
    )
    resampled_differences = difference_per_match[sampled_matches].sum(axis=1) / shots_per_match[
        sampled_matches
    ].sum(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(resampled_differences, [tail, 1.0 - tail])
    return LogLossDifference(
        mean=float(shot_differences.mean()), lower=float(lower), upper=float(upper)
    )


def save_calibration_plot(
    shots_by_split: dict[str, pd.DataFrame],
    probabilities_by_model: dict[str, dict[str, FloatArray]],
    n_bins: int,
    path: Path,
) -> None:
    figure = Figure(figsize=(5.5 * len(shots_by_split), 5))
    axes = figure.subplots(1, len(shots_by_split), sharex=True, sharey=True, squeeze=False)[0]
    for axis, (split_name, split_shots) in zip(axes, shots_by_split.items(), strict=True):
        axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfectly calibrated")
        for model_name, probabilities_by_split in probabilities_by_model.items():
            observed_rate, predicted_rate = calibration_curve(
                split_shots[columns.GOAL],
                probabilities_by_split[split_name],
                n_bins=n_bins,
                strategy="quantile",
            )
            axis.plot(predicted_rate, observed_rate, marker="o", markersize=4, label=model_name)
        axis.set_title(f"{split_name} (n={len(split_shots)})")
        axis.set_xlabel("mean predicted goal probability")
        axis.set_ylabel("observed goal rate")
        axis.legend(fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
