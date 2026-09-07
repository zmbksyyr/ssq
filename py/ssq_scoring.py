"""Compatibility facade for lottery-ball scoring."""

from ssq_ball_scoring import BallScoringDependencies
from ssq_ball_scoring import (
    run_strategy_and_get_scores as _run_strategy_and_get_scores,
)
from ssq_config import validate_strategy_params
from ssq_features import validate_feature_columns
from ssq_parsing import validate_ball_scores
from ssq_score_adjustments import apply_red_score_adjustments
from ssq_score_signals import get_omission, get_weighted_frequency
from ssq_training import predict_positive_probability, validate_model_sets


def run_strategy_and_get_scores(
    df_history,
    params,
    ml_models_red,
    ml_models_blue,
    feature_columns,
):
    """Run score fusion through patchable legacy module boundaries."""
    dependencies = BallScoringDependencies(
        validate_params=validate_strategy_params,
        validate_models=validate_model_sets,
        validate_features=validate_feature_columns,
        predict_probability=predict_positive_probability,
        weighted_frequency=get_weighted_frequency,
        omission=get_omission,
        adjust_red_scores=apply_red_score_adjustments,
        validate_scores=validate_ball_scores,
    )
    return _run_strategy_and_get_scores(
        df_history,
        params,
        ml_models_red,
        ml_models_blue,
        feature_columns,
        dependencies,
    )
