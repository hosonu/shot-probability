import logging
from pathlib import Path

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shot_probability import columns
from shot_probability.config import TrainConfig
from shot_probability.evaluate import FloatArray, save_calibration_plot, score_predictions
from shot_probability.runtime import configure_logging, set_seed
from shot_probability.split import split_by_match

logger = logging.getLogger(__name__)

PRIOR_MODEL_NAME = "prior"
LOGISTIC_MODEL_NAME = "logistic"


def load_shots(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make data` first.")
    return pd.read_parquet(path, columns=columns.SHOT_TABLE_COLUMNS)


def build_logistic_pipeline(config: TrainConfig) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), columns.NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), columns.CATEGORICAL_FEATURES),
            ("boolean", "passthrough", columns.BOOLEAN_FEATURES),
        ]
    )
    classifier = LogisticRegression(
        C=config.regularization_strength, max_iter=config.max_iter, random_state=config.seed
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def predict_goal_probability(model: ClassifierMixin, shots: pd.DataFrame) -> FloatArray:
    return model.predict_proba(shots[columns.FEATURES])[:, 1]  # type: ignore[no-any-return]


def run_training(config: TrainConfig) -> pd.DataFrame:
    set_seed(config.seed)
    shots = load_shots(config.shots_path)
    split = split_by_match(shots, config.validation_fraction, config.test_fraction, config.seed)
    logger.info(
        "Shots: train=%d validation=%d test=%d",
        len(split.train),
        len(split.validation),
        len(split.test),
    )

    models: dict[str, ClassifierMixin] = {
        PRIOR_MODEL_NAME: DummyClassifier(strategy="prior"),
        LOGISTIC_MODEL_NAME: build_logistic_pipeline(config),
    }
    for model in models.values():
        model.fit(split.train[columns.FEATURES], split.train[columns.GOAL])

    evaluated_splits = {"validation": split.validation, "test": split.test}
    metric_rows = [
        score_predictions(
            model_name,
            split_name,
            split_shots[columns.GOAL],
            predict_goal_probability(model, split_shots),
        )
        for model_name, model in models.items()
        for split_name, split_shots in evaluated_splits.items()
    ]
    metrics = pd.DataFrame(metric_rows)

    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(config.outputs_dir / "metrics.csv", index=False)
    save_calibration_plot(
        evaluated_splits,
        {
            split_name: predict_goal_probability(models[LOGISTIC_MODEL_NAME], split_shots)
            for split_name, split_shots in evaluated_splits.items()
        },
        config.calibration_bins,
        config.outputs_dir / "calibration.png",
    )
    return metrics


def main() -> None:
    configure_logging()
    metrics = run_training(TrainConfig())
    logger.info("Metrics:\n%s", metrics.to_string(index=False))


if __name__ == "__main__":
    main()
