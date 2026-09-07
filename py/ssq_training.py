"""LightGBM model specifications, training, and prediction validation."""

import lightgbm as lgb
from ssq_domain import BLUE_BALLS, RED_BALLS
from ssq_features import validate_feature_columns
from ssq_model_contracts import MODEL_TRAINING_PARAMS, BallModelSpec
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
    """Train one binary next-draw model per candidate."""
    if not isinstance(spec, BallModelSpec):
        raise TypeError('spec 必须为 BallModelSpec')
    models = {}
    iterator = (
        tqdm(spec.candidates, desc=spec.description, ncols=80)
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
        model = lgb.LGBMClassifier(**MODEL_TRAINING_PARAMS)
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
    """Train the complete red and blue model sets."""
    feature_columns = validate_feature_columns(training_df, feature_columns)
    red_models = train_models_for_spec(
        training_df,
        feature_columns,
        BallModelSpec(
            candidates=RED_BALLS,
            outcome_column='红球',
            contains_candidate=lambda draw, ball: ball in draw,
            description='训练红球模型' if show_progress else None,
        ),
    )
    blue_models = train_models_for_spec(
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
