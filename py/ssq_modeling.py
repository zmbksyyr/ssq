"""Compatibility facade for features, training, and strategy scoring."""

import ssq_features as _features
import ssq_scoring as _scoring
import ssq_training as _training
from ssq_core import validate_ball_scores  # noqa: F401 - compatibility export

FEATURE_COLUMNS = _features.FEATURE_COLUMNS
feature_engineer = _features.feature_engineer
validate_feature_columns = _features.validate_feature_columns

MODEL_TRAINING_PARAMS = _training.MODEL_TRAINING_PARAMS
BallModelSpec = _training.BallModelSpec
train_models_for_spec = _training.train_models_for_spec
train_ball_models = _training.train_ball_models
train_prediction_models = _training.train_prediction_models
validate_model_sets = _training.validate_model_sets
predict_positive_probability = _training.predict_positive_probability
lgb = _training.lgb

get_omission = _scoring.get_omission
get_weighted_frequency = _scoring.get_weighted_frequency
apply_red_score_adjustments = _scoring.apply_red_score_adjustments
run_strategy_and_get_scores = _scoring.run_strategy_and_get_scores
