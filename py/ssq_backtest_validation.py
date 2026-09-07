"""Validation of public rolling-backtest controls."""

from ssq_config import RED_POOL_MODES, StrategyConfig, normalize_integer_param


def validate_backtest_request(num_periods, pool_modes, config):
    """Normalize and validate public backtest controls before expensive work."""
    num_periods = normalize_integer_param('num_periods', num_periods)
    if num_periods < 0:
        raise ValueError('num_periods 不能为负数')
    if isinstance(pool_modes, str):
        raise TypeError('pool_modes 必须为候选池模式序列，不能是字符串')
    try:
        pool_modes = tuple(pool_modes)
    except TypeError as exc:
        raise TypeError('pool_modes 必须为候选池模式序列') from exc
    if not pool_modes:
        raise ValueError('pool_modes 不能为空')
    if len(pool_modes) != len(set(pool_modes)):
        raise ValueError('pool_modes 不能包含重复模式')
    invalid_modes = sorted(set(pool_modes) - set(RED_POOL_MODES))
    if invalid_modes:
        raise ValueError(f'未知候选池模式: {invalid_modes}')
    if not isinstance(config, StrategyConfig):
        raise TypeError('config 必须为 StrategyConfig')
    return num_periods, pool_modes
