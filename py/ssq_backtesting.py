"""Historical rule audits and leakage-free rolling strategy backtests."""

from collections import Counter

import ssq_backtest_evaluation as _backtest_evaluation
import ssq_backtest_models as _backtest_models
import ssq_backtest_preparation as _backtest_preparation
import ssq_backtest_validation as _backtest_validation
from ssq_anti_crowding import make_rejection_set, rejection_seed_for_issue
from ssq_backtest_metrics import BacktestAccumulator, BacktestResult
from ssq_candidates import count_actual_reds_by_rank_band
from ssq_config import (
    DEFAULT_STRATEGY_CONFIG,
    RULE_AUDIT_PERIODS,
    validate_strategy_params,
)
from ssq_rank_bands import build_rank_band_widths
from ssq_rule_auditing import FILTER_NAMES as _FILTER_NAMES
from ssq_rule_auditing import (
    audit_historical_hard_pipeline as _audit_historical_hard_pipeline,
)
from ssq_rule_auditing import (
    audit_historical_rule_coverage as _audit_historical_rule_coverage,
)
from ssq_rule_auditing import historical_rule_context as _historical_rule_context
from ssq_scoring import run_strategy_and_get_scores
from ssq_training import train_prediction_models, validate_model_sets
from tqdm import tqdm

FILTER_NAMES = _FILTER_NAMES
BacktestIssue = _backtest_models.BacktestIssue
BacktestSelectionInputs = _backtest_models.BacktestSelectionInputs
BacktestRunContext = _backtest_models.BacktestRunContext
BacktestRequest = _backtest_models.BacktestRequest
PreparedBacktestIssue = _backtest_models.PreparedBacktestIssue
evaluate_backtest_mode = _backtest_evaluation.evaluate_backtest_mode
record_backtest_selection = _backtest_evaluation.record_backtest_selection
validate_backtest_request = _backtest_validation.validate_backtest_request


def historical_rule_context(full_df, index):
    """Compatibility wrapper for the rule-auditing context builder."""
    return _historical_rule_context(full_df, index)


def audit_historical_rule_coverage(full_df, periods=RULE_AUDIT_PERIODS):
    """Compatibility wrapper for independent historical rule coverage."""
    return _audit_historical_rule_coverage(full_df, periods)


def audit_historical_hard_pipeline(full_df, periods=RULE_AUDIT_PERIODS):
    """Compatibility wrapper for cumulative hard-rule coverage."""
    return _audit_historical_hard_pipeline(full_df, periods)


def prepare_backtest_issue(full_df, index, run_context):
    """Compatibility wrapper for leakage-free issue preparation."""
    dependencies = _backtest_preparation.BacktestPreparationDependencies(
        train_models=train_prediction_models,
        validate_models=validate_model_sets,
        score_balls=run_strategy_and_get_scores,
        count_rank_bands=count_actual_reds_by_rank_band,
        build_rule_context=historical_rule_context,
        derive_rejection_seed=rejection_seed_for_issue,
        build_rejection_set=make_rejection_set,
    )
    return _backtest_preparation.prepare_backtest_issue(
        full_df,
        index,
        run_context,
        dependencies,
    )


def run_backtest(full_df, request):
    """Run a rolling backtest that retrains using only earlier draws."""
    if not isinstance(request, BacktestRequest):
        raise TypeError('request 必须为 BacktestRequest')
    num_periods, pool_modes = validate_backtest_request(
        request.num_periods,
        request.pool_modes,
        request.config,
    )
    params = validate_strategy_params(request.params)
    feature_columns = request.feature_columns
    config = request.config
    print('\n' + '=' * 70)
    print(f'        最近 {num_periods} 期完整策略滚动回测')
    print('=' * 70)

    minimum_history = num_periods + 50
    if len(full_df) < minimum_history:
        print(f'历史数据不足 {minimum_history} 期，无法执行回测。跳过此步骤。')
        return {
            mode: BacktestResult(
                0, 0, 0, 0, 0, Counter(),
                rank_band_widths=build_rank_band_widths(config),
            )
            for mode in pool_modes
        }

    rank_band_widths = build_rank_band_widths(config)
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

    with tqdm(total=len(backtest_range), desc='执行严谨回测', ncols=80) as progress:
        for offset, index in enumerate(backtest_range):
            prepared = prepare_backtest_issue(full_df, index, run_context)
            if prepared is None:
                progress.update(1)
                continue
            window_name = 'earlier' if offset < split_offset else 'recent'

            for mode in pool_modes:
                evaluate_backtest_mode(
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


def run_full_backtest(
    full_df,
    params,
    feature_columns,
    num_periods,
    pool_modes=('mixed',),
    config=DEFAULT_STRATEGY_CONFIG,
):
    """Compatibility wrapper for the request-based backtest API."""
    return run_backtest(
        full_df,
        BacktestRequest(
            params=params,
            feature_columns=feature_columns,
            num_periods=num_periods,
            pool_modes=pool_modes,
            config=config,
        ),
    )
