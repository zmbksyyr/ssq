"""Dependency-driven training of per-ball prediction models."""

from collections.abc import Callable
from dataclasses import dataclass

import lightgbm as lgb
from ssq_domain import BLUE_BALLS, RED_BALLS
from ssq_features import validate_feature_columns
from ssq_model_contracts import MODEL_TRAINING_PARAMS, BallModelSpec
from tqdm import tqdm


@dataclass(frozen=True)
class ModelFitDependencies:
    classifier_factory: Callable
    progress_factory: Callable


@dataclass(frozen=True)
class PredictionTrainingDependencies:
    validate_features: Callable
    train_spec: Callable


def default_fit_dependencies():
    return ModelFitDependencies(
        classifier_factory=lgb.LGBMClassifier,
        progress_factory=tqdm,
    )


def default_prediction_dependencies():
    return PredictionTrainingDependencies(
        validate_features=validate_feature_columns,
        train_spec=train_models_for_spec,
    )


def train_models_for_spec(
    training_df,
    feature_columns,
    spec,
    dependencies=None,
):
    """Train one binary next-draw model per candidate."""
    if not isinstance(spec, BallModelSpec):
        raise TypeError('spec 必须为 BallModelSpec')
    dependencies = dependencies or default_fit_dependencies()
    models = {}
    iterator = (
        dependencies.progress_factory(
            spec.candidates,
            desc=spec.description,
            ncols=80,
        )
        if spec.description else spec.candidates
    )
    features = training_df[list(feature_columns)]
    for candidate in iterator:
        target = training_df[spec.outcome_column].apply(
            lambda outcome, current=candidate: int(
                spec.contains_candidate(outcome, current)
            )
        ).shift(-1)
        valid_rows = target.notna() & features.notna().all(axis=1)
        if not valid_rows.any():
            continue
        model = dependencies.classifier_factory(**MODEL_TRAINING_PARAMS)
        model.fit(features.loc[valid_rows], target.loc[valid_rows])
        models[candidate] = model
    return models


def train_ball_models(
    training_df,
    feature_columns,
    candidates,
    outcome_column,
    contains_candidate,
    description=None,
    dependencies=None,
):
    """Train a caller-defined family of per-ball binary models."""
    return train_models_for_spec(
        training_df,
        feature_columns,
        BallModelSpec(
            candidates=candidates,
            outcome_column=outcome_column,
            contains_candidate=contains_candidate,
            description=description,
        ),
        dependencies,
    )


def train_prediction_models(
    training_df,
    feature_columns,
    show_progress=False,
    dependencies=None,
):
    """Train the complete red and blue model sets."""
    dependencies = dependencies or default_prediction_dependencies()
    feature_columns = dependencies.validate_features(training_df, feature_columns)
    red_models = dependencies.train_spec(
        training_df,
        feature_columns,
        BallModelSpec(
            candidates=RED_BALLS,
            outcome_column='红球',
            contains_candidate=lambda draw, ball: ball in draw,
            description='训练红球模型' if show_progress else None,
        ),
    )
    blue_models = dependencies.train_spec(
        training_df,
        feature_columns,
        BallModelSpec(
            candidates=BLUE_BALLS,
            outcome_column='蓝球',
            contains_candidate=lambda drawn, ball: drawn == ball,
            description='训练蓝球模型' if show_progress else None,
        ),
    )
    return red_models, blue_models
