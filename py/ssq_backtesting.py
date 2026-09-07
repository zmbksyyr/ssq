"""Historical rule audits and leakage-free rolling strategy backtests."""

import ssq_backtest_evaluation as _backtest_evaluation
import ssq_backtest_metrics as _backtest_metrics
import ssq_backtest_models as _backtest_models
import ssq_backtest_preparation as _backtest_preparation
import ssq_backtest_runner as _backtest_runner
import ssq_backtest_validation as _backtest_validation
import ssq_candidates as _candidates
from ssq_anti_crowding import make_rejection_set, rejection_seed_for_issue
from ssq_ball_scoring import run_strategy_and_get_scores
from ssq_config import (
    DEFAULT_STRATEGY_CONFIG,
    RULE_AUDIT_PERIODS,
    validate_strategy_params,
)
from ssq_model_training import train_prediction_models
from ssq_model_validation import validate_model_sets
from ssq_rank_bands import build_rank_band_widths
from ssq_red_pool import count_actual_reds_by_rank_band
from ssq_rule_auditing import FILTER_NAMES as _FILTER_NAMES
from ssq_rule_auditing import (
    audit_historical_hard_pipeline as _audit_historical_hard_pipeline,
)
from ssq_rule_auditing import (
    audit_historical_rule_coverage as _audit_historical_rule_coverage,
)
from ssq_rule_auditing import historical_rule_context as _historical_rule_context
from tqdm import tqdm

FILTER_NAMES = _FILTER_NAMES
BacktestAccumulator = _backtest_metrics.BacktestAccumulator
BacktestResult = _backtest_metrics.BacktestResult
BacktestIssue = _backtest_models.BacktestIssue
BacktestSelectionInputs = _backtest_models.BacktestSelectionInputs
BacktestRunContext = _backtest_models.BacktestRunContext
BacktestRequest = _backtest_models.BacktestRequest
PreparedBacktestIssue = _backtest_models.PreparedBacktestIssue
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


def evaluate_backtest_mode(
    mode,
    current,
    issue,
    selection_inputs,
    additional_accumulators=(),
):
    """Compatibility wrapper binding patchable candidate generation."""
    dependencies = _backtest_evaluation.BacktestEvaluationDependencies(
        generate_candidates=_candidates.generate_candidates,
    )
    return _backtest_evaluation.evaluate_backtest_mode(
        mode,
        current,
        issue,
        selection_inputs,
        additional_accumulators,
        dependencies,
    )


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
    """Compatibility wrapper binding legacy patchable dependencies."""
    dependencies = _backtest_runner.BacktestRunnerDependencies(
        validate_request=validate_backtest_request,
        validate_params=validate_strategy_params,
        build_rank_band_widths=build_rank_band_widths,
        prepare_issue=prepare_backtest_issue,
        evaluate_mode=evaluate_backtest_mode,
        progress_factory=tqdm,
    )
    return _backtest_runner.run_backtest(full_df, request, dependencies)


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
