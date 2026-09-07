"""Dependency-driven execution of leakage-free rolling backtests."""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ssq_backtest_evaluation import evaluate_backtest_mode
from ssq_backtest_metrics import BacktestAccumulator, BacktestResult
from ssq_backtest_models import BacktestRequest, BacktestRunContext
from ssq_backtest_preparation import prepare_backtest_issue
from ssq_backtest_validation import validate_backtest_request
from ssq_config import validate_strategy_params
from ssq_rank_bands import build_rank_band_widths
from tqdm import tqdm


@dataclass(frozen=True)
class BacktestRunnerDependencies:
    validate_request: Callable[..., Any]
    validate_params: Callable[..., Any]
    build_rank_band_widths: Callable[..., dict]
    prepare_issue: Callable[..., Any]
    evaluate_mode: Callable[..., Any]
    progress_factory: Callable[..., Any]


def default_runner_dependencies():
    return BacktestRunnerDependencies(
        validate_request=validate_backtest_request,
        validate_params=validate_strategy_params,
        build_rank_band_widths=build_rank_band_widths,
        prepare_issue=prepare_backtest_issue,
        evaluate_mode=evaluate_backtest_mode,
        progress_factory=tqdm,
    )


def run_backtest(full_df, request, dependencies=None):
    """Run a rolling backtest that retrains using only earlier draws."""
    if not isinstance(request, BacktestRequest):
        raise TypeError('request 必须为 BacktestRequest')
    dependencies = dependencies or default_runner_dependencies()
    num_periods, pool_modes = dependencies.validate_request(
        request.num_periods,
        request.pool_modes,
        request.config,
    )
    params = dependencies.validate_params(request.params)
    feature_columns = request.feature_columns
    config = request.config
    print('\n' + '=' * 70)
    print(f'        最近 {num_periods} 期完整策略滚动回测')
    print('=' * 70)

    minimum_history = num_periods + 50
    rank_band_widths = dependencies.build_rank_band_widths(config)
    if len(full_df) < minimum_history:
        print(f'历史数据不足 {minimum_history} 期，无法执行回测。跳过此步骤。')
        return {
            mode: BacktestResult(
                0, 0, 0, 0, 0, Counter(),
                rank_band_widths=rank_band_widths.copy(),
            )
            for mode in pool_modes
        }

    metrics = {
        mode: BacktestAccumulator(rank_band_widths=rank_band_widths.copy())
        for mode in pool_modes
    }
    window_metrics = {
        name: {
            mode: BacktestAccumulator(rank_band_widths=rank_band_widths.copy())
            for mode in pool_modes
        }
        for name in ('earlier', 'recent')
    }
    backtest_range = range(len(full_df) - num_periods, len(full_df))
    split_offset = len(backtest_range) // 2
    window_periods = {
        'earlier': split_offset,
        'recent': len(backtest_range) - split_offset,
    }
    run_context = BacktestRunContext(params, feature_columns, config)

    with dependencies.progress_factory(
        total=len(backtest_range),
        desc='执行严谨回测',
        ncols=80,
    ) as progress:
        for offset, index in enumerate(backtest_range):
            prepared = dependencies.prepare_issue(full_df, index, run_context)
            if prepared is None:
                progress.update(1)
                continue
            window_name = 'earlier' if offset < split_offset else 'recent'

            for mode in pool_modes:
                dependencies.evaluate_mode(
                    mode,
                    metrics[mode],
                    prepared.issue,
                    prepared.selection_inputs,
                    additional_accumulators=(
                        window_metrics[window_name][mode],
                    ),
                )
            progress.update(1)

    print('回测完成。\n')
    return {
        mode: value.to_result(len(backtest_range), windows=(
            {
                name: window_metrics[name][mode].to_result(window_periods[name])
                for name in ('earlier', 'recent')
            }
            if len(backtest_range) >= 2 else {}
        ))
        for mode, value in metrics.items()
    }
