"""LightGBM model specifications, training, and prediction validation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import lightgbm as lgb
import numpy as np
from ssq_domain import BLUE_BALLS, RED_BALLS
from ssq_features import validate_feature_columns
from tqdm import tqdm

MODEL_TRAINING_PARAMS = MappingProxyType({
    'random_state': 42,
    'deterministic': True,
    'force_col_wise': True,
    'verbose': -1,
})


@dataclass(frozen=True)
class BallModelSpec:
    candidates: Sequence[int]
    outcome_column: str
    contains_candidate: Callable[[object, int], bool]
    description: str | None = None


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
            raise ValueError(f'单类别模型只能包含类别 0 或 1: {classes.tolist()}')
        return np.full(len(features), float(classes[0] == 1))
    positive_columns = np.flatnonzero(classes == 1)
    if len(positive_columns) != 1:
        raise ValueError(f'模型类别缺少唯一正类 1: {classes.tolist()}')
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
