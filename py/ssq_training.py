"""Compatibility facade for per-ball model training and validation."""

import lightgbm as lgb
from ssq_features import validate_feature_columns
from ssq_model_contracts import MODEL_TRAINING_PARAMS, BallModelSpec
from ssq_model_training import (
    ModelFitDependencies,
    PredictionTrainingDependencies,
)
from ssq_model_training import train_models_for_spec as _train_models_for_spec
from ssq_model_training import train_prediction_models as _train_prediction_models
from ssq_model_validation import predict_positive_probability, validate_model_sets
from tqdm import tqdm

__all__ = [
    'MODEL_TRAINING_PARAMS',
    'BallModelSpec',
    'predict_positive_probability',
    'train_ball_models',
    'train_models_for_spec',
    'train_prediction_models',
    'validate_model_sets',
]


def train_models_for_spec(training_df, feature_columns, spec):
    """Train one model family through patchable legacy boundaries."""
    dependencies = ModelFitDependencies(
        classifier_factory=lgb.LGBMClassifier,
        progress_factory=tqdm,
    )
    return _train_models_for_spec(
        training_df,
        feature_columns,
        spec,
        dependencies,
    )


def train_ball_models(
    training_df,
    feature_columns,
    candidates,
    outcome_column,
    contains_candidate,
    description=None,
):
    """Compatibility wrapper for the specification-based training API."""
    return train_models_for_spec(
        training_df,
        feature_columns,
        BallModelSpec(
            candidates=candidates,
            outcome_column=outcome_column,
            contains_candidate=contains_candidate,
            description=description,
        ),
    )


def train_prediction_models(training_df, feature_columns, show_progress=False):
    """Train red and blue model sets through patchable legacy boundaries."""
    dependencies = PredictionTrainingDependencies(
        validate_features=validate_feature_columns,
        train_spec=train_models_for_spec,
    )
    return _train_prediction_models(
        training_df,
        feature_columns,
        show_progress,
        dependencies,
    )
