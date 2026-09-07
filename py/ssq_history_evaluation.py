"""Historical rule auditing and rolling backtest workflow stage."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ssq_backtest_models import BacktestRequest
from ssq_backtest_runner import run_backtest
from ssq_rule_auditing import (
    audit_historical_hard_pipeline,
    audit_historical_rule_coverage,
)
from ssq_strategy_params import load_strategy_params
from ssq_workflow_models import HistoricalEvaluation


@dataclass(frozen=True)
class HistoryEvaluationDependencies:
    load_params: Callable[..., Any]
    audit_rules: Callable[..., dict]
    audit_hard_pipeline: Callable[..., dict]
    run_backtest: Callable[..., dict]


def default_evaluation_dependencies():
    return HistoryEvaluationDependencies(
        load_params=load_strategy_params,
        audit_rules=audit_historical_rule_coverage,
        audit_hard_pipeline=audit_historical_hard_pipeline,
        run_backtest=run_backtest,
    )


def evaluate_history(history, options, params_path, dependencies=None):
    """Load parameters, audit rules, and backtest one prepared history."""
    if dependencies is None:
        dependencies = default_evaluation_dependencies()
    try:
        loaded_params = dependencies.load_params(params_path)
    except ValueError as exc:
        raise SystemExit(f'错误: {exc}') from exc
    if not loaded_params.loaded_from_file:
        print(f'警告: 未找到参数文件 {params_path}，将使用内置的默认参数。')

    rule_coverage = dependencies.audit_rules(
        history.frame,
        options.rule_audit_periods,
    )
    hard_pipeline_coverage = dependencies.audit_hard_pipeline(
        history.frame,
        options.rule_audit_periods,
    )
    backtests = dependencies.run_backtest(
        history.frame,
        BacktestRequest(
            params=loaded_params.values,
            feature_columns=history.feature_columns,
            num_periods=options.backtest_periods,
            pool_modes=options.backtest_pool_modes,
            config=options.strategy_config,
        ),
    )
    return HistoricalEvaluation(
        loaded_params=loaded_params,
        rule_coverage=rule_coverage,
        hard_pipeline_coverage=hard_pipeline_coverage,
        backtests=backtests,
        selected_backtest=backtests[options.pool_mode],
    )
