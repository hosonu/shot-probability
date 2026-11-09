from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shot_probability import columns
from shot_probability.columns import FeatureSet
from shot_probability.config import BenchmarkConfig, GradientBoostingConfig
from shot_probability.geometry import FloatArray


class ShotModel(Protocol):
    """The single interface every benchmarked model implements.

    `validation` may be used for early stopping or model selection only. The test split is never
    passed to `fit`.
    """

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> None: ...

    def predict_goal_probability(self, shots: pd.DataFrame) -> FloatArray: ...


def build_preprocessing(feature_set: FeatureSet) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), list(feature_set.numeric)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(feature_set.categorical),
            ),
            ("boolean", "passthrough", list(feature_set.boolean)),
        ]
    )


class PriorModel:
    """Predicts the training goal rate for every shot; the floor every model must beat."""

    def __init__(self) -> None:
        self._goal_rate = float("nan")

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> None:
        self._goal_rate = float(train[columns.GOAL].mean())

    def predict_goal_probability(self, shots: pd.DataFrame) -> FloatArray:
        return np.full(len(shots), self._goal_rate)


class SklearnShotModel:
    def __init__(self, pipeline: Pipeline, feature_set: FeatureSet) -> None:
        self._pipeline = pipeline
        self._feature_names = feature_set.names

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> None:
        self._pipeline.fit(train[self._feature_names], train[columns.GOAL])

    def predict_goal_probability(self, shots: pd.DataFrame) -> FloatArray:
        return self._pipeline.predict_proba(shots[self._feature_names])[:, 1]  # type: ignore[no-any-return]


class GradientBoostingShotModel(SklearnShotModel):
    """Boosting rounds are chosen on the validation split, then the model is refit on train."""

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> None:
        super().fit(train, validation)
        validation_features = self._pipeline[:-1].transform(validation[self._feature_names])
        staged_probabilities = self._pipeline[-1].staged_predict_proba(validation_features)
        validation_losses = [
            log_loss(validation[columns.GOAL], probabilities[:, 1], labels=[0, 1])
            for probabilities in staged_probabilities
        ]
        best_iteration_count = int(np.argmin(validation_losses)) + 1
        self._pipeline.set_params(classifier__max_iter=best_iteration_count)
        super().fit(train, validation)

    @property
    def iteration_count(self) -> int:
        return int(self._pipeline[-1].n_iter_)


def build_logistic_model(config: BenchmarkConfig, feature_set: FeatureSet) -> SklearnShotModel:
    classifier = LogisticRegression(
        C=config.regularization_strength, max_iter=config.max_iter, random_state=config.seed
    )
    pipeline = Pipeline(
        [("preprocessing", build_preprocessing(feature_set)), ("classifier", classifier)]
    )
    return SklearnShotModel(pipeline, feature_set)


def build_gradient_boosting_model(
    config: BenchmarkConfig, feature_set: FeatureSet
) -> GradientBoostingShotModel:
    boosting: GradientBoostingConfig = config.gradient_boosting
    classifier = HistGradientBoostingClassifier(
        learning_rate=boosting.learning_rate,
        max_depth=boosting.max_depth,
        min_samples_leaf=boosting.min_samples_leaf,
        l2_regularization=boosting.l2_regularization,
        max_iter=boosting.max_iterations,
        early_stopping=False,
        random_state=config.seed,
    )
    pipeline = Pipeline(
        [("preprocessing", build_preprocessing(feature_set)), ("classifier", classifier)]
    )
    return GradientBoostingShotModel(pipeline, feature_set)
