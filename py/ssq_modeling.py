"""Feature engineering, model training, and ball score calculation."""

from collections import Counter

import numpy as np
import pandas as pd
import ssq_features as _features
import ssq_training as _training
from ssq_config import validate_strategy_params
from ssq_core import (
    BLUE_BALLS,
    RED_BALLS,
    validate_ball_scores,
)

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


def get_omission(df):
    """Calculate how many draws each red ball has been absent."""
    total_draws = len(df)
    last_positions = {}
    for position, draw in enumerate(df['红球']):
        for ball in draw:
            last_positions[ball] = position
    return {
        ball: total_draws - last_positions[ball] - 1
        if ball in last_positions else total_draws
        for ball in RED_BALLS
    }


def get_weighted_frequency(series, decay_factor):
    """Calculate frequency with exponentially greater weight on recent draws."""
    draw_count = len(series)
    weights = np.array([
        decay_factor ** (draw_count - index - 1)
        for index in range(draw_count)
    ])
    weighted_counts = {}
    for index, numbers in enumerate(series):
        for ball in numbers:
            weighted_counts[ball] = weighted_counts.get(ball, 0) + weights[index]
    return pd.Series(weighted_counts)


def apply_red_score_adjustments(red_scores, df_history, params):
    """Apply the hot, cold, and previous-draw bonuses."""
    adjusted = dict(red_scores)
    hot_lookback = int(params.get('hot_lookback', 0))
    hot_threshold = int(params.get('hot_threshold', 0))
    if hot_lookback > 0 and hot_threshold > 0:
        recent = df_history.tail(hot_lookback)['红球']
        hot_counts = Counter(ball for draw in recent for ball in draw)
        for ball, count in hot_counts.items():
            if count >= hot_threshold:
                adjusted[ball] *= params.get('hot_bonus', 1.0)

    cold_lookback = int(params.get('cold_lookback', 0))
    if cold_lookback > 0:
        recent_numbers = {
            ball for draw in df_history.tail(cold_lookback)['红球'] for ball in draw
        }
        for ball in set(RED_BALLS) - recent_numbers:
            adjusted[ball] *= params.get('cold_bonus', 1.0)

    if not df_history.empty:
        for ball in df_history.iloc[-1]['红球']:
            adjusted[ball] *= params.get('repeat_bonus', 1.0)
    return adjusted


def run_strategy_and_get_scores(
    df_history,
    params,
    ml_models_red,
    ml_models_blue,
    feature_columns,
):
    """Combine frequency, omission, and model signals into ball scores."""
    params = validate_strategy_params(params)
    validate_model_sets(ml_models_red, ml_models_blue)
    feature_columns = validate_feature_columns(df_history, feature_columns)
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

    red_weighted_freq = get_weighted_frequency(
        df_history['红球'], params['decay_factor']
    )
    red_omission = get_omission(df_history)
    red_ml_probs = {
        ball: predict_positive_probability(ml_models_red[ball], last_features)[0]
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
    red_scores = apply_red_score_adjustments(red_scores, df_history, params)
    red_scores = validate_ball_scores(red_scores, RED_BALLS, '红球')

    blue_weighted_freq = get_weighted_frequency(
        df_history['蓝球'].apply(lambda ball: [ball]), params['decay_factor']
    )
    blue_ml_probs = {
        ball: predict_positive_probability(ml_models_blue[ball], last_features)[0]
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
    blue_scores = validate_ball_scores(blue_scores, BLUE_BALLS, '蓝球')
    return red_scores, blue_scores
