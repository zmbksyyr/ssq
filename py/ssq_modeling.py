"""Feature engineering, model training, and ball score calculation."""

from collections import Counter
from itertools import combinations, pairwise

import lightgbm as lgb
import numpy as np
import pandas as pd
from ssq_core import (
    BLUE_BALLS,
    PRIME_RED_BALLS,
    RED_BALLS,
    validate_ball_scores,
)
from tqdm import tqdm

FEATURE_COLUMNS = (
    'red_sum',
    'red_span',
    'odd_count',
    'blue_lag1',
    'red_zone_small',
    'red_zone_medium',
    'red_zone_large',
    'red_big_count',
    'red_prime_count',
    'red_sum_tail',
    'red_consecutive_groups',
    'red_ac_value',
    'red_tail_uniques',
    'red_sum_lag1',
    'odd_count_lag1',
    'red_sum_ma5',
    'odd_count_ma5',
    'blue_ma5',
)


def _count_consecutive_groups(numbers):
    groups = 0
    in_group = False
    for current, following in pairwise(numbers):
        if following - current == 1:
            if not in_group:
                groups += 1
                in_group = True
        else:
            in_group = False
    return groups


def feature_engineer(df):
    """Return a copy of draw history with the declared model input features."""
    df = df.copy()
    df['red_sum'] = df['红球'].apply(sum)
    df['red_span'] = df['红球'].apply(lambda numbers: max(numbers) - min(numbers))
    df['odd_count'] = df['红球'].apply(
        lambda numbers: sum(number % 2 != 0 for number in numbers)
    )
    df['blue_lag1'] = df['蓝球'].shift(1)
    df['red_zone_small'] = df['红球'].apply(
        lambda numbers: sum(1 <= number <= 11 for number in numbers)
    )
    df['red_zone_medium'] = df['红球'].apply(
        lambda numbers: sum(12 <= number <= 22 for number in numbers)
    )
    df['red_zone_large'] = df['红球'].apply(
        lambda numbers: sum(23 <= number <= 33 for number in numbers)
    )
    df['red_big_count'] = df['红球'].apply(
        lambda numbers: sum(number > 16 for number in numbers)
    )
    df['red_prime_count'] = df['红球'].apply(
        lambda numbers: sum(number in PRIME_RED_BALLS for number in numbers)
    )
    df['red_sum_tail'] = df['red_sum'].apply(lambda value: value % 10)
    df['red_consecutive_groups'] = df['红球'].apply(_count_consecutive_groups)
    df['red_ac_value'] = df['红球'].apply(
        lambda numbers: len({
            abs(first - second) for first, second in combinations(numbers, 2)
        })
    )
    df['red_tail_uniques'] = df['红球'].apply(
        lambda numbers: len({number % 10 for number in numbers})
    )

    window_size = 5
    df['red_sum_lag1'] = df['red_sum'].shift(1)
    df['odd_count_lag1'] = df['odd_count'].shift(1)
    df['red_sum_ma5'] = df['red_sum'].shift(1).rolling(window=window_size).mean()
    df['odd_count_ma5'] = df['odd_count'].shift(1).rolling(window=window_size).mean()
    df['blue_ma5'] = df['蓝球'].shift(1).rolling(window=window_size).mean()
    return df


def validate_feature_columns(frame, feature_columns=FEATURE_COLUMNS):
    """Reject missing, duplicate, or undeclared model feature names."""
    columns = tuple(feature_columns)
    if len(columns) != len(set(columns)):
        raise ValueError('模型特征列不能重复')
    missing = sorted(set(FEATURE_COLUMNS) - set(columns))
    undeclared = sorted(set(columns) - set(FEATURE_COLUMNS))
    if missing:
        raise ValueError(f'缺少声明的模型特征列: {missing}')
    if undeclared:
        raise ValueError(f'包含未声明的模型特征列: {undeclared}')
    if columns != FEATURE_COLUMNS:
        raise ValueError('模型特征列顺序与声明不一致')
    missing_from_frame = sorted(set(columns) - set(frame.columns))
    if missing_from_frame:
        raise ValueError(f'模型数据缺少特征列: {missing_from_frame}')
    return columns


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


def train_ball_models(
    training_df,
    feature_columns,
    candidates,
    outcome_column,
    contains_candidate,
    description=None,
):
    """Train one binary next-draw model per candidate."""
    models = {}
    iterator = (
        tqdm(candidates, desc=description, ncols=80) if description else candidates
    )
    features = training_df[list(feature_columns)]
    for candidate in iterator:
        target = training_df[outcome_column].apply(
            lambda outcome, current=candidate: int(
                contains_candidate(outcome, current)
            )
        ).shift(-1)
        valid_rows = target.notna() & features.notna().all(axis=1)
        if not valid_rows.any():
            continue
        model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        model.fit(features.loc[valid_rows], target.loc[valid_rows])
        models[candidate] = model
    return models


def train_prediction_models(training_df, feature_columns, show_progress=False):
    """Train the complete red and blue model sets."""
    feature_columns = validate_feature_columns(training_df, feature_columns)
    red_models = train_ball_models(
        training_df,
        feature_columns,
        RED_BALLS,
        '红球',
        lambda draw, ball: ball in draw,
        '训练红球模型' if show_progress else None,
    )
    blue_models = train_ball_models(
        training_df,
        feature_columns,
        BLUE_BALLS,
        '蓝球',
        lambda drawn, ball: drawn == ball,
        '训练蓝球模型' if show_progress else None,
    )
    return red_models, blue_models


def validate_model_sets(red_models, blue_models):
    missing_red = sorted(set(RED_BALLS) - set(red_models))
    missing_blue = sorted(set(BLUE_BALLS) - set(blue_models))
    extra_red = sorted(set(red_models) - set(RED_BALLS))
    extra_blue = sorted(set(blue_models) - set(BLUE_BALLS))
    if missing_red or missing_blue or extra_red or extra_blue:
        raise ValueError(
            '模型集合与号码范围不一致: '
            f'红球缺失 {missing_red}, 红球多余 {extra_red}, '
            f'蓝球缺失 {missing_blue}, 蓝球多余 {extra_blue}'
        )


def predict_positive_probability(model, features):
    """Return P(class=1), including correct behavior for one-class models."""
    classes = np.asarray(model.classes_)
    if len(classes) == 1:
        if classes[0] not in (0, 1):
            raise ValueError(f"单类别模型只能包含类别 0 或 1: {classes.tolist()}")
        return np.full(len(features), float(classes[0] == 1))
    positive_columns = np.flatnonzero(classes == 1)
    if len(positive_columns) != 1:
        raise ValueError(f"模型类别缺少唯一正类 1: {classes.tolist()}")
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    expected_shape = (len(features), len(classes))
    if probabilities.shape != expected_shape:
        raise ValueError(
            f'模型概率矩阵形状应为 {expected_shape}, 实际为 {probabilities.shape}'
        )
    positive = probabilities[:, int(positive_columns[0])]
    if not np.isfinite(positive).all() or ((positive < 0) | (positive > 1)).any():
        raise ValueError('模型正类概率必须是 0 到 1 之间的有限数值')
    return positive


def run_strategy_and_get_scores(
    df_history,
    params,
    ml_models_red,
    ml_models_blue,
    feature_columns,
):
    """Combine frequency, omission, and model signals into ball scores."""
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
