"""Validation of trained model sets and positive-class probabilities."""

import numpy as np
from ssq_domain import BLUE_BALLS, RED_BALLS


def validate_model_sets(red_models, blue_models):
    """Require exactly one trained model for every red and blue ball."""
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
