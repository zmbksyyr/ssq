"""Defaults, normalization, and validation for strategy parameters."""

import numpy as np

DEFAULT_PARAMS = {
    'decay_factor': 0.999,
    'weight_freq': 0.4,
    'weight_omission': 0.5,
    'weight_ml': 0.1,
    'hot_lookback': 10,
    'hot_threshold': 2,
    'hot_bonus': 1.2,
    'cold_lookback': 30,
    'cold_bonus': 1.05,
    'repeat_bonus': 1.15,
    'weight_blue_freq': 0.6,
    'weight_blue_ml': 0.4,
}
INTEGER_PARAM_NAMES = ('hot_lookback', 'hot_threshold', 'cold_lookback')
FLOAT_PARAM_NAMES = (
    'decay_factor', 'weight_freq', 'weight_omission', 'weight_ml',
    'hot_bonus', 'cold_bonus', 'repeat_bonus',
    'weight_blue_freq', 'weight_blue_ml',
)


def normalize_integer_param(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} 必须为整数')
    return int(value)


def normalize_float_param(name, value):
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f'{name} 必须为数字')
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f'{name} 必须为有限数值')
    return normalized


def validate_weight_group(params, names):
    values = [params[name] for name in names]
    if any(value < 0 for value in values) or not np.isclose(sum(values), 1.0):
        raise ValueError(f"权重 {', '.join(names)} 必须非负且总和为 1")


def validate_param_ranges(params):
    if not 0 < params['decay_factor'] <= 1:
        raise ValueError('decay_factor 必须在 (0, 1] 范围内')
    for name in INTEGER_PARAM_NAMES:
        if params[name] < 0:
            raise ValueError(f'{name} 不能为负数')
    for name in ('hot_bonus', 'cold_bonus', 'repeat_bonus'):
        if params[name] <= 0:
            raise ValueError(f'{name} 必须大于 0')


def validate_strategy_params(params):
    if not isinstance(params, dict):
        raise TypeError('strategy params must be a JSON object')
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise ValueError(f"未知策略参数: {', '.join(unknown)}")
    merged = {**DEFAULT_PARAMS, **params}
    for name in INTEGER_PARAM_NAMES:
        merged[name] = normalize_integer_param(name, merged[name])
    for name in FLOAT_PARAM_NAMES:
        merged[name] = normalize_float_param(name, merged[name])
    for group in (
        ('weight_freq', 'weight_omission', 'weight_ml'),
        ('weight_blue_freq', 'weight_blue_ml'),
    ):
        validate_weight_group(merged, group)
    validate_param_ranges(merged)
    return merged
