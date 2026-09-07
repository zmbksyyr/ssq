"""Command-line parsing for the SSQ analyzer."""

import argparse

from ssq_config_models import (
    BACKTEST_PERIODS,
    RANDOM_SEED,
    RED_POOL_MODES,
    REJECTION_LIB_SIZE,
    RULE_AUDIT_PERIODS,
    AnalyzerOptions,
)


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
