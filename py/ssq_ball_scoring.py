"""Dependency-driven fusion of frequency, omission, and model signals."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from ssq_config import validate_strategy_params
from ssq_domain import BLUE_BALLS, RED_BALLS
from ssq_features import validate_feature_columns
from ssq_parsing import validate_ball_scores
from ssq_score_adjustments import apply_red_score_adjustments
from ssq_score_signals import get_omission, get_weighted_frequency
from ssq_training import predict_positive_probability, validate_model_sets


@dataclass(frozen=True)
class BallScoringDependencies:
    """Replaceable calculations used by score fusion."""

    validate_params: Callable[..., dict] = validate_strategy_params
    validate_models: Callable[..., None] = validate_model_sets
    validate_features: Callable[..., tuple] = validate_feature_columns
    predict_probability: Callable[..., Any] = predict_positive_probability
    weighted_frequency: Callable[..., Any] = get_weighted_frequency
    omission: Callable[..., dict] = get_omission
    adjust_red_scores: Callable[..., dict] = apply_red_score_adjustments
    validate_scores: Callable[..., dict] = validate_ball_scores


def run_strategy_and_get_scores(
    df_history,
    params,
    ml_models_red,
    ml_models_blue,
    feature_columns,
    dependencies=None,
):
    """Combine frequency, omission, and model signals into ball scores."""
    dependencies = dependencies or BallScoringDependencies()
    params = dependencies.validate_params(params)
    dependencies.validate_models(ml_models_red, ml_models_blue)
    feature_columns = dependencies.validate_features(df_history, feature_columns)
    if df_history.empty:
        raise ValueError('评分历史数据不能为空')

    last_features = df_history.iloc[[-1]][list(feature_columns)].copy()
    for column in last_features.columns:
        if last_features[column].isnull().any():
            last_features[column] = last_features[column].fillna(
                df_history[column].mean()
            )
    try:
        feature_values = last_features.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('最新一期模型特征必须为数值') from exc
    if not np.isfinite(feature_values).all():
        raise ValueError('最新一期模型特征包含无法填补的非有限值')

    red_weighted_freq = dependencies.weighted_frequency(
        df_history['红球'], params['decay_factor']
    )
    red_omission = dependencies.omission(df_history)
    red_ml_probs = {
        ball: dependencies.predict_probability(
            ml_models_red[ball], last_features
        )[0]
        for ball in RED_BALLS
    }
    max_red_freq = red_weighted_freq.max() or 1
    max_red_omission = max(red_omission.values()) or 1
    red_scores = {
        ball: (
            red_weighted_freq.get(ball, 0) / max_red_freq * params['weight_freq']
            + red_omission.get(ball, 0) / max_red_omission
            * params['weight_omission']
            + red_ml_probs[ball] * params['weight_ml']
        )
        for ball in RED_BALLS
    }
    red_scores = dependencies.adjust_red_scores(red_scores, df_history, params)
    red_scores = dependencies.validate_scores(red_scores, RED_BALLS, '红球')

    blue_weighted_freq = dependencies.weighted_frequency(
        df_history['蓝球'].apply(lambda ball: [ball]), params['decay_factor']
    )
    blue_ml_probs = {
        ball: dependencies.predict_probability(
            ml_models_blue[ball], last_features
        )[0]
        for ball in BLUE_BALLS
    }
    max_blue_freq = blue_weighted_freq.max() or 1
    blue_scores = {
        ball: (
            blue_weighted_freq.get(ball, 0) / max_blue_freq
            * params['weight_blue_freq']
            + blue_ml_probs[ball] * params['weight_blue_ml']
        )
        for ball in BLUE_BALLS
    }
    blue_scores = dependencies.validate_scores(blue_scores, BLUE_BALLS, '蓝球')
    return red_scores, blue_scores
