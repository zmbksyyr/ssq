"""Strategy defaults, validation, and analyzer command-line options."""

import argparse

import ssq_config_models as _config_models
import ssq_strategy_validation as _strategy_validation

POOL_SIZE_RED = _config_models.POOL_SIZE_RED
RED_HIGH_COUNT = _config_models.RED_HIGH_COUNT
RED_LOW_COUNT = _config_models.RED_LOW_COUNT
NUM_BLUE_BALLS = _config_models.NUM_BLUE_BALLS
REJECTION_LIB_SIZE = _config_models.REJECTION_LIB_SIZE
RANDOM_SEED = _config_models.RANDOM_SEED
REJECTION_SEED_MULTIPLIER = _config_models.REJECTION_SEED_MULTIPLIER
BACKTEST_PERIODS = _config_models.BACKTEST_PERIODS
INTERACTIVE_THRESHOLD = _config_models.INTERACTIVE_THRESHOLD
COUNTDOWN_SECONDS = _config_models.COUNTDOWN_SECONDS
NUM_RECOMMENDATIONS = _config_models.NUM_RECOMMENDATIONS
MAX_SHARED_RED_BALLS = _config_models.MAX_SHARED_RED_BALLS
RULE_AUDIT_PERIODS = _config_models.RULE_AUDIT_PERIODS
TOTAL_RED_COMBINATIONS = _config_models.TOTAL_RED_COMBINATIONS
RED_POOL_MODES = _config_models.RED_POOL_MODES

DEFAULT_PARAMS = _strategy_validation.DEFAULT_PARAMS
INTEGER_PARAM_NAMES = _strategy_validation.INTEGER_PARAM_NAMES
FLOAT_PARAM_NAMES = _strategy_validation.FLOAT_PARAM_NAMES
normalize_integer_param = _strategy_validation.normalize_integer_param
normalize_float_param = _strategy_validation.normalize_float_param
validate_weight_group = _strategy_validation.validate_weight_group
validate_param_ranges = _strategy_validation.validate_param_ranges
validate_strategy_params = _strategy_validation.validate_strategy_params


StrategyConfig = _config_models.StrategyConfig
DEFAULT_STRATEGY_CONFIG = _config_models.DEFAULT_STRATEGY_CONFIG
AnalyzerOptions = _config_models.AnalyzerOptions
LoadedStrategyParams = _config_models.LoadedStrategyParams


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
