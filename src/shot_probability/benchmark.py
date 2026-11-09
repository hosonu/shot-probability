import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from shot_probability import columns
from shot_probability.config import BenchmarkConfig
from shot_probability.evaluate import (
    FloatArray,
    bootstrap_log_loss_difference,
    save_calibration_plot,
    score_predictions,
)
from shot_probability.models import (
    PriorModel,
    ShotModel,
    build_gradient_boosting_model,
    build_logistic_model,
)
from shot_probability.runtime import configure_logging, set_seed
from shot_probability.sequence_model import TransformerShotModel
from shot_probability.split import split_by_match

logger = logging.getLogger(__name__)

PRIOR_MODEL_NAME = "prior"
BASELINE_MODEL_NAME = "logistic"

ProbabilitiesByModel = dict[str, dict[str, FloatArray]]


@dataclass(frozen=True)
class BenchmarkResult:
    metrics: pd.DataFrame
    comparison_with_baseline: pd.DataFrame


def load_benchmark_table(
    shots_path: Path, context_path: Path, freeze_frame_path: Path
) -> pd.DataFrame:
    for path in (shots_path, context_path, freeze_frame_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `make data` first.")
    shots = pd.read_parquet(shots_path, columns=columns.SHOT_TABLE_COLUMNS)
    context = pd.read_parquet(context_path)
    freeze_frame = pd.read_parquet(freeze_frame_path)
    benchmark_table = shots.merge(
        context, on=[columns.MATCH_ID, columns.EVENT_ID], how="left", validate="one_to_one"
    ).merge(
        freeze_frame, on=[columns.MATCH_ID, columns.EVENT_ID], how="left", validate="one_to_one"
    )
    missing_context_count = int(benchmark_table[columns.N_PRIOR_EVENTS].isna().sum())
    if missing_context_count:
        raise ValueError(f"{missing_context_count} shots have no pre-shot context row")
    missing_freeze_frame_count = int(benchmark_table[columns.HAS_FREEZE_FRAME].isna().sum())
    if missing_freeze_frame_count:
        raise ValueError(f"{missing_freeze_frame_count} shots have no freeze-frame row")
    return benchmark_table


def build_models(config: BenchmarkConfig) -> dict[str, ShotModel]:
    return {
        PRIOR_MODEL_NAME: PriorModel(),
        BASELINE_MODEL_NAME: build_logistic_model(config, columns.SHOT_FEATURES),
        "logistic_context": build_logistic_model(config, columns.CONTEXT_FEATURES),
        "gradient_boosting": build_gradient_boosting_model(config, columns.SHOT_FEATURES),
        "gradient_boosting_context": build_gradient_boosting_model(
            config, columns.CONTEXT_FEATURES
        ),
        "logistic_freeze_frame": build_logistic_model(config, columns.FREEZE_FRAME_FEATURES),
        "gradient_boosting_freeze_frame": build_gradient_boosting_model(
            config, columns.FREEZE_FRAME_FEATURES
        ),
        "transformer": TransformerShotModel(config.transformer, config.seed),
    }


def predict_all(
    models: dict[str, ShotModel], shots_by_split: dict[str, pd.DataFrame]
) -> ProbabilitiesByModel:
    return {
        model_name: {
            split_name: model.predict_goal_probability(split_shots)
            for split_name, split_shots in shots_by_split.items()
        }
        for model_name, model in models.items()
    }


def score_all(
    probabilities_by_model: ProbabilitiesByModel, shots_by_split: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            score_predictions(
                model_name, split_name, shots_by_split[split_name][columns.GOAL], probabilities
            )
            for model_name, probabilities_by_split in probabilities_by_model.items()
            for split_name, probabilities in probabilities_by_split.items()
        ]
    )


def compare_with_baseline(
    probabilities_by_model: ProbabilitiesByModel,
    shots_by_split: dict[str, pd.DataFrame],
    config: BenchmarkConfig,
) -> pd.DataFrame:
    excluded_names = {PRIOR_MODEL_NAME, BASELINE_MODEL_NAME}
    rows = []
    for model_name, probabilities_by_split in probabilities_by_model.items():
        if model_name in excluded_names:
            continue
        for split_name, split_shots in shots_by_split.items():
            difference = bootstrap_log_loss_difference(
                split_shots[columns.GOAL],
                split_shots[columns.MATCH_ID],
                probabilities_by_model[BASELINE_MODEL_NAME][split_name],
                probabilities_by_split[split_name],
                config.bootstrap_resamples,
                config.confidence_level,
                config.seed,
            )
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "log_loss_difference": difference.mean,
                    "ci_lower": difference.lower,
                    "ci_upper": difference.upper,
                }
            )
    return pd.DataFrame(rows)


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    set_seed(config.seed)
    benchmark_table = load_benchmark_table(
        config.shots_path, config.context_path, config.freeze_frame_path
    )
    split = split_by_match(
        benchmark_table, config.validation_fraction, config.test_fraction, config.seed
    )
    logger.info(
        "Shots: train=%d validation=%d test=%d",
        len(split.train),
        len(split.validation),
        len(split.test),
    )

    models = build_models(config)
    for model_name, model in models.items():
        logger.info("Fitting %s", model_name)
        model.fit(split.train, split.validation)

    shots_by_split = {"validation": split.validation, "test": split.test}
    probabilities_by_model = predict_all(models, shots_by_split)
    result = BenchmarkResult(
        metrics=score_all(probabilities_by_model, shots_by_split),
        comparison_with_baseline=compare_with_baseline(
            probabilities_by_model, shots_by_split, config
        ),
    )

    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(config.outputs_dir / "metrics.csv", index=False)
    result.comparison_with_baseline.to_csv(
        config.outputs_dir / "log_loss_vs_logistic.csv", index=False
    )
    save_calibration_plot(
        shots_by_split,
        {
            model_name: probabilities_by_split
            for model_name, probabilities_by_split in probabilities_by_model.items()
            if model_name != PRIOR_MODEL_NAME
        },
        config.calibration_bins,
        config.outputs_dir / "calibration.png",
    )
    return result


def main() -> None:
    configure_logging()
    result = run_benchmark(BenchmarkConfig())
    logger.info("Metrics:\n%s", result.metrics.to_string(index=False))
    logger.info(
        "Log loss difference vs %s (negative is better):\n%s",
        BASELINE_MODEL_NAME,
        result.comparison_with_baseline.to_string(index=False),
    )


if __name__ == "__main__":
    main()
