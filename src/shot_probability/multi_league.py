"""Experiments on the four-league table: learning curve and information-group comparison.

The single-league benchmark in ``benchmark.py`` is published and left untouched; everything
here reads ``data/multi_league/`` and writes ``outputs/multi_league/``.
"""

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from shot_probability import columns
from shot_probability.benchmark import load_benchmark_table
from shot_probability.build_multi_league import MULTI_LEAGUE_SHOT_COLUMNS
from shot_probability.columns import FeatureSet, combine_feature_sets
from shot_probability.config import (
    BenchmarkConfig,
    MultiLeagueDataConfig,
    MultiLeagueExperimentConfig,
)
from shot_probability.evaluate import (
    FloatArray,
    bootstrap_log_loss_difference,
    per_shot_log_loss,
)
from shot_probability.models import (
    ShotModel,
    build_gradient_boosting_model,
    build_logistic_model,
)
from shot_probability.runtime import configure_logging, set_seed
from shot_probability.split import split_by_match, split_leave_one_league_out, subsample_matches

logger = logging.getLogger(__name__)

LOGISTIC = "logistic"
GRADIENT_BOOSTING = "gradient_boosting"
MODEL_BUILDERS: dict[str, Callable[[BenchmarkConfig, FeatureSet], ShotModel]] = {
    LOGISTIC: build_logistic_model,
    GRADIENT_BOOSTING: build_gradient_boosting_model,
}
STATSBOMB_REFERENCE = "statsbomb_xg_reference"
POOLED_SCOPE = "all leagues"
BASE_SET = "base"
ALL_SET = "all"

LEARNING_CURVE_FEATURE_SETS: dict[str, FeatureSet] = {
    "shot": columns.SHOT_FEATURES,
    "shot+context+freeze_frame": combine_feature_sets(
        columns.SHOT_FEATURES, columns.CONTEXT_GROUP, columns.FREEZE_FRAME_GROUP
    ),
}


@dataclass(frozen=True)
class Comparison:
    """Candidate minus baseline log loss; ``kind`` is ``add``, ``remove`` or ``combination``."""

    kind: str
    group: str
    candidate: str
    baseline: str


def load_multi_league_table(config: MultiLeagueDataConfig) -> pd.DataFrame:
    if not config.attributes_path.exists():
        raise FileNotFoundError(f"{config.attributes_path} not found. Run `make data-multi` first.")
    table = load_benchmark_table(
        config.shots_path,
        config.context_path,
        config.freeze_frame_path,
        shot_columns=MULTI_LEAGUE_SHOT_COLUMNS,
    ).merge(
        pd.read_parquet(config.attributes_path),
        on=[columns.MATCH_ID, columns.EVENT_ID],
        how="left",
        validate="one_to_one",
    )
    missing_attribute_count = int(table[columns.IS_PENALTY].isna().sum())
    if missing_attribute_count:
        raise ValueError(f"{missing_attribute_count} shots have no shot-attribute row")
    return table


def fit_and_predict(
    model_name: str,
    feature_set: FeatureSet,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: BenchmarkConfig,
) -> FloatArray:
    model = MODEL_BUILDERS[model_name](config, feature_set)
    model.fit(train, validation)
    return model.predict_goal_probability(test)


# --- learning curve ---------------------------------------------------------------------------


def run_learning_curve(table: pd.DataFrame, config: MultiLeagueExperimentConfig) -> pd.DataFrame:
    """Logistic vs gradient boosting test log loss as the training set grows.

    One match-level split pooled over all leagues; validation and test stay fixed while the
    training set is a nested, seeded subsample of whole matches.
    """
    model_config = config.model
    split = split_by_match(
        table, model_config.validation_fraction, model_config.test_fraction, model_config.seed
    )
    targets = [
        target for target in config.learning_curve_train_shots if target < len(split.train)
    ] + [len(split.train)]
    logger.info(
        "Learning curve: train pool=%d validation=%d test=%d targets=%s",
        len(split.train),
        len(split.validation),
        len(split.test),
        targets,
    )
    rows = []
    for feature_set_name, feature_set in LEARNING_CURVE_FEATURE_SETS.items():
        for target in targets:
            train = subsample_matches(split.train, target, model_config.seed)
            probabilities = {
                model_name: fit_and_predict(
                    model_name, feature_set, train, split.validation, split.test, model_config
                )
                for model_name in MODEL_BUILDERS
            }
            difference = bootstrap_log_loss_difference(
                split.test[columns.GOAL],
                split.test[columns.MATCH_ID],
                probabilities[LOGISTIC],
                probabilities[GRADIENT_BOOSTING],
                model_config.bootstrap_resamples,
                model_config.confidence_level,
                model_config.seed,
            )
            goals = split.test[columns.GOAL].to_numpy(dtype=float)
            rows.append(
                {
                    "feature_set": feature_set_name,
                    "train_shots": len(train),
                    "train_matches": train[columns.MATCH_ID].nunique(),
                    "logistic_test_log_loss": per_shot_log_loss(
                        goals, probabilities[LOGISTIC]
                    ).mean(),
                    "gradient_boosting_test_log_loss": per_shot_log_loss(
                        goals, probabilities[GRADIENT_BOOSTING]
                    ).mean(),
                    "gb_minus_logistic": difference.mean,
                    "ci_lower": difference.lower,
                    "ci_upper": difference.upper,
                }
            )
            logger.info("Learning curve row: %s", rows[-1])
    return pd.DataFrame(rows)


def choose_primary_model(learning_curve: pd.DataFrame) -> str:
    """Gradient boosting only if, at the largest training set, it beats logistic regression
    with the 95% interval below zero on every feature set; otherwise logistic regression."""
    largest = learning_curve.loc[learning_curve.groupby("feature_set")["train_shots"].idxmax()]
    return GRADIENT_BOOSTING if bool((largest["ci_upper"] < 0).all()) else LOGISTIC


def save_learning_curve_plot(learning_curve: pd.DataFrame, path: Path) -> None:
    feature_set_names = list(dict.fromkeys(learning_curve["feature_set"]))
    figure = Figure(figsize=(5.5 * len(feature_set_names), 4.5))
    axes = figure.subplots(1, len(feature_set_names), squeeze=False)[0]
    for axis, feature_set_name in zip(axes, feature_set_names, strict=True):
        rows = learning_curve.loc[learning_curve["feature_set"] == feature_set_name]
        for model_name in MODEL_BUILDERS:
            axis.plot(
                rows["train_shots"],
                rows[f"{model_name}_test_log_loss"],
                marker="o",
                label=model_name,
            )
        axis.set_title(feature_set_name)
        axis.set_xlabel("training shots")
        axis.set_ylabel("test log loss (lower is better)")
        axis.legend(fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


# --- information-group comparison ------------------------------------------------------------


def build_group_feature_sets(groups: dict[str, FeatureSet]) -> dict[str, FeatureSet]:
    """Base, base + each group, base + context + freeze frame, everything, everything - each."""
    base = columns.SHOT_FEATURES
    feature_sets = {BASE_SET: base}
    for group_name, group in groups.items():
        feature_sets[f"+{group_name}"] = combine_feature_sets(base, group)
    if {"context", "freeze_frame"} <= groups.keys():
        feature_sets["+context+freeze_frame"] = combine_feature_sets(
            base, groups["context"], groups["freeze_frame"]
        )
    feature_sets[ALL_SET] = combine_feature_sets(base, *groups.values())
    for group_name in groups:
        others = [group for name, group in groups.items() if name != group_name]
        feature_sets[f"-{group_name}"] = combine_feature_sets(base, *others)
    return feature_sets


def build_comparisons(groups: dict[str, FeatureSet]) -> list[Comparison]:
    comparisons = [
        Comparison("add", group_name, f"+{group_name}", BASE_SET) for group_name in groups
    ]
    if {"context", "freeze_frame"} <= groups.keys():
        comparisons.append(
            Comparison("combination", "context", "+context+freeze_frame", "+freeze_frame")
        )
    comparisons += [
        Comparison("remove", group_name, f"-{group_name}", ALL_SET) for group_name in groups
    ]
    return comparisons


def predict_leave_one_league_out(
    table: pd.DataFrame,
    feature_sets: dict[str, FeatureSet],
    model_names: tuple[str, ...],
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, dict[tuple[str, str], FloatArray]]:
    """Out-of-league predictions: every shot is predicted once, by models that never saw its
    league. Returns the concatenated test shots and probabilities aligned with them."""
    leagues = list(dict.fromkeys(table[columns.LEAGUE]))
    test_tables = []
    probabilities: dict[tuple[str, str], list[FloatArray]] = {
        (model_name, set_name): [] for model_name in model_names for set_name in feature_sets
    }
    for league in leagues:
        split = split_leave_one_league_out(table, league, config.validation_fraction, config.seed)
        logger.info(
            "Held-out %s: train=%d validation=%d test=%d",
            league,
            len(split.train),
            len(split.validation),
            len(split.test),
        )
        test_tables.append(split.test)
        for (model_name, set_name), fold_probabilities in probabilities.items():
            fold_probabilities.append(
                fit_and_predict(
                    model_name,
                    feature_sets[set_name],
                    split.train,
                    split.validation,
                    split.test,
                    config,
                )
            )
    return pd.concat(test_tables, ignore_index=True), {
        key: np.concatenate(fold_probabilities) for key, fold_probabilities in probabilities.items()
    }


def score_out_of_league(
    test_shots: pd.DataFrame, probabilities: dict[tuple[str, str], FloatArray]
) -> pd.DataFrame:
    """Log loss per model and feature set, pooled and per held-out league, with the StatsBomb
    xG reference row (never an input, scored on the same shots)."""
    goals = test_shots[columns.GOAL].to_numpy(dtype=float)
    losses = {
        (model_name, set_name): per_shot_log_loss(goals, model_probabilities)
        for (model_name, set_name), model_probabilities in probabilities.items()
    }
    losses[(STATSBOMB_REFERENCE, "-")] = per_shot_log_loss(
        goals, test_shots[columns.STATSBOMB_XG].to_numpy(dtype=float)
    )
    rows = []
    for (model_name, set_name), shot_losses in losses.items():
        per_league = pd.Series(shot_losses).groupby(test_shots[columns.LEAGUE].to_numpy()).mean()
        for scope, log_loss_value in [(POOLED_SCOPE, shot_losses.mean()), *per_league.items()]:
            rows.append(
                {
                    "model": model_name,
                    "feature_set": set_name,
                    "test_league": scope,
                    "log_loss": float(log_loss_value),
                }
            )
    return pd.DataFrame(rows)


def compare_out_of_league(
    test_shots: pd.DataFrame,
    probabilities: dict[tuple[str, str], FloatArray],
    comparisons: list[Comparison],
    model_names: tuple[str, ...],
    config: BenchmarkConfig,
) -> pd.DataFrame:
    """Paired match-level bootstrap of candidate minus baseline, pooled and per league."""
    scopes = [(POOLED_SCOPE, np.ones(len(test_shots), dtype=bool))] + [
        (league, (test_shots[columns.LEAGUE] == league).to_numpy())
        for league in dict.fromkeys(test_shots[columns.LEAGUE])
    ]
    rows = []
    for model_name in model_names:
        for comparison in comparisons:
            for scope, mask in scopes:
                difference = bootstrap_log_loss_difference(
                    test_shots.loc[mask, columns.GOAL],
                    test_shots.loc[mask, columns.MATCH_ID],
                    probabilities[(model_name, comparison.baseline)][mask],
                    probabilities[(model_name, comparison.candidate)][mask],
                    config.bootstrap_resamples,
                    config.confidence_level,
                    config.seed,
                )
                rows.append(
                    {
                        "model": model_name,
                        "kind": comparison.kind,
                        "group": comparison.group,
                        "candidate": comparison.candidate,
                        "baseline": comparison.baseline,
                        "test_league": scope,
                        "log_loss_difference": difference.mean,
                        "ci_lower": difference.lower,
                        "ci_upper": difference.upper,
                    }
                )
    return pd.DataFrame(rows)


def run_group_comparison(
    table: pd.DataFrame, groups: dict[str, FeatureSet], config: MultiLeagueExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_sets = build_group_feature_sets(groups)
    model_names = (config.primary_model, config.check_model)
    test_shots, probabilities = predict_leave_one_league_out(
        table, feature_sets, model_names, config.model
    )
    return (
        score_out_of_league(test_shots, probabilities),
        compare_out_of_league(
            test_shots, probabilities, build_comparisons(groups), model_names, config.model
        ),
    )


# --- entry point -------------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="experiment", required=True)
    subparsers.add_parser("learning-curve", help="logistic vs gradient boosting by data size")
    groups_parser = subparsers.add_parser("groups", help="add / remove information groups")
    groups_parser.add_argument(
        "--groups",
        default=",".join(columns.FEATURE_GROUPS),
        help="comma-separated subset of: " + ", ".join(columns.FEATURE_GROUPS),
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    arguments = parse_arguments()
    config = MultiLeagueExperimentConfig()
    set_seed(config.model.seed)
    table = load_multi_league_table(config.data)
    config.outputs_dir.mkdir(parents=True, exist_ok=True)

    if arguments.experiment == "learning-curve":
        learning_curve = run_learning_curve(table, config)
        learning_curve.to_csv(config.outputs_dir / "learning_curve.csv", index=False)
        save_learning_curve_plot(learning_curve, config.outputs_dir / "learning_curve.png")
        logger.info("Learning curve:\n%s", learning_curve.to_string(index=False))
        logger.info(
            "Primary model by the pre-registered rule: %s", choose_primary_model(learning_curve)
        )
        return

    group_names = [name.strip() for name in arguments.groups.split(",") if name.strip()]
    unknown = set(group_names) - columns.FEATURE_GROUPS.keys()
    if unknown:
        raise ValueError(f"Unknown groups: {sorted(unknown)}")
    groups = {name: columns.FEATURE_GROUPS[name] for name in group_names}
    metrics, comparison = run_group_comparison(table, groups, config)
    stage_dir = config.outputs_dir / ("groups_" + "_".join(group_names))
    stage_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(stage_dir / "metrics.csv", index=False)
    comparison.to_csv(stage_dir / "comparison.csv", index=False)
    logger.info("Primary model: %s, check model: %s", config.primary_model, config.check_model)
    logger.info(
        "Pooled out-of-league log loss:\n%s",
        metrics.loc[metrics["test_league"] == POOLED_SCOPE].to_string(index=False),
    )
    logger.info(
        "Comparisons (candidate - baseline, negative is better):\n%s",
        comparison.loc[comparison["test_league"] == POOLED_SCOPE].to_string(index=False),
    )


if __name__ == "__main__":
    main()
