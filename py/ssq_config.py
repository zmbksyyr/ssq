"""Strategy defaults, validation, and analyzer command-line options."""

import argparse
from dataclasses import dataclass
from math import comb

import numpy as np
from ssq_core import BLUE_BALLS, RED_BALLS, RED_COUNT

POOL_SIZE_RED = 17
RED_HIGH_COUNT = 4
RED_LOW_COUNT = 4
NUM_BLUE_BALLS = 7

REJECTION_LIB_SIZE = 500_000
RANDOM_SEED = 42
REJECTION_SEED_MULTIPLIER = 1_000_000_007

BACKTEST_PERIODS = 200
INTERACTIVE_THRESHOLD = 100
COUNTDOWN_SECONDS = 10
NUM_RECOMMENDATIONS = 10
MAX_SHARED_RED_BALLS = 4
RULE_AUDIT_PERIODS = 200
TOTAL_RED_COMBINATIONS = comb(len(RED_BALLS), RED_COUNT)
RED_POOL_MODES = ('mixed', 'high', 'middle', 'low')

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


@dataclass(frozen=True)
class StrategyConfig:
    """Selection settings shared by backtests and live runs."""

    pool_size_red: int = POOL_SIZE_RED
    high_count: int = RED_HIGH_COUNT
    low_count: int = RED_LOW_COUNT
    blue_count: int = NUM_BLUE_BALLS
    recommendation_count: int = NUM_RECOMMENDATIONS
    max_shared_red_balls: int = MAX_SHARED_RED_BALLS
    rejection_lib_size: int = REJECTION_LIB_SIZE
    random_seed: int = RANDOM_SEED

    def __post_init__(self):
        integer_fields = (
            'pool_size_red', 'high_count', 'low_count', 'blue_count',
            'recommendation_count', 'max_shared_red_balls',
            'rejection_lib_size', 'random_seed',
        )
        for name in integer_fields:
            object.__setattr__(
                self,
                name,
                normalize_integer_param(name, getattr(self, name)),
            )
        if not RED_COUNT <= self.pool_size_red <= len(RED_BALLS):
            raise ValueError('pool_size_red must be between 6 and 33')
        if min(self.high_count, self.low_count) < 0:
            raise ValueError('high_count and low_count cannot be negative')
        if self.high_count + self.low_count > self.pool_size_red:
            raise ValueError('high_count and low_count exceed pool_size_red')
        if not 1 <= self.blue_count <= len(BLUE_BALLS):
            raise ValueError('blue_count must be between 1 and 16')
        if self.recommendation_count < 1:
            raise ValueError('recommendation_count must be positive')
        if not 0 <= self.max_shared_red_balls <= RED_COUNT:
            raise ValueError('max_shared_red_balls must be between 0 and 6')
        if not 0 <= self.rejection_lib_size <= TOTAL_RED_COMBINATIONS:
            raise ValueError(
                f'rejection_lib_size must be between 0 and {TOTAL_RED_COMBINATIONS}'
            )


DEFAULT_STRATEGY_CONFIG = StrategyConfig()


@dataclass(frozen=True)
class AnalyzerOptions:
    backtest_periods: int = BACKTEST_PERIODS
    rejection_size: int = REJECTION_LIB_SIZE
    seed: int = RANDOM_SEED
    pool_mode: str = 'mixed'
    compare_pools: bool = False
    non_interactive: bool = False
    rule_audit_periods: int = RULE_AUDIT_PERIODS

    def __post_init__(self):
        integer_fields = (
            'backtest_periods', 'rejection_size', 'seed', 'rule_audit_periods',
        )
        for name in integer_fields:
            object.__setattr__(
                self,
                name,
                normalize_integer_param(name, getattr(self, name)),
            )
        for name in ('compare_pools', 'non_interactive'):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f'{name} 必须为布尔值')
        if self.backtest_periods < 0 or self.rule_audit_periods < 0:
            raise ValueError('backtest-periods 和 rule-audit-periods 不能为负数')
        if not 0 <= self.rejection_size <= TOTAL_RED_COMBINATIONS:
            raise ValueError(
                f'rejection-size 必须在 0 到 {TOTAL_RED_COMBINATIONS} 之间'
            )
        if self.pool_mode not in RED_POOL_MODES:
            raise ValueError(f'未知候选池模式: {self.pool_mode}')

    @property
    def strategy_config(self):
        return StrategyConfig(
            rejection_lib_size=self.rejection_size,
            random_seed=self.seed,
        )

    @property
    def backtest_pool_modes(self):
        return RED_POOL_MODES if self.compare_pools else (self.pool_mode,)


@dataclass(frozen=True)
class LoadedStrategyParams:
    values: dict
    loaded_from_file: bool


def build_argument_parser():
    parser = argparse.ArgumentParser(description='双色球策略分析与推荐')
    parser.add_argument(
        '--backtest-periods', type=int, default=BACKTEST_PERIODS,
        help='回测期数，默认 %(default)s',
    )
    parser.add_argument(
        '--rejection-size', type=int, default=REJECTION_LIB_SIZE,
        help='反撞号随机库大小，默认 %(default)s',
    )
    parser.add_argument(
        '--seed', type=int, default=RANDOM_SEED,
        help='随机种子，默认 %(default)s',
    )
    parser.add_argument(
        '--pool-mode', choices=RED_POOL_MODES, default='mixed',
        help='红球候选池模式，默认 %(default)s',
    )
    parser.add_argument(
        '--compare-pools', action='store_true',
        help='在一次回测中对比四种候选池模式',
    )
    parser.add_argument(
        '--non-interactive', action='store_true',
        help='不等待键盘输入，适用于自动化运行',
    )
    parser.add_argument(
        '--rule-audit-periods', type=int, default=RULE_AUDIT_PERIODS,
        help='统计规则对真实开奖覆盖率的期数，默认 %(default)s',
    )
    return parser


def parse_cli_options(argv=None):
    parser = build_argument_parser()
    namespace = parser.parse_args(argv)
    try:
        return AnalyzerOptions(**vars(namespace))
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
