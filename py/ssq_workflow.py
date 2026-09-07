"""Application workflow for loading data, running analysis, and saving reports."""

import ssq_history_evaluation as _history_evaluation
import ssq_history_preparation as _history_preparation
import ssq_paths as _paths
import ssq_prediction_workflow as _prediction_workflow
import ssq_report_output as _report_output
import ssq_strategy_params as _strategy_params
import ssq_workflow_models as _workflow_models
from ssq_analysis_result import build_analysis_report_data
from ssq_anti_crowding import make_rejection_set, rejection_seed_for_issue
from ssq_backtest_runner import run_backtest
from ssq_candidates import generate_candidates
from ssq_config import (
    parse_cli_options,
)
from ssq_console import (
    display_passed_combinations as _display_passed_combinations,
)
from ssq_console import (
    get_user_input_with_timeout as _get_user_input_with_timeout,
)
from ssq_console import (
    is_confirmation_input as _is_confirmation_input,
)
from ssq_draw_data import (
    fingerprint_draw_frame,
    normalize_draw_frame,
    validate_draw_dates_not_future,
)
from ssq_draw_schedule import infer_next_issue, local_now
from ssq_duplex import rank_duplex_candidates
from ssq_features import feature_engineer
from ssq_file_io import atomic_write_text
from ssq_report_builder import build_analysis_report
from ssq_rule_auditing import (
    audit_historical_hard_pipeline,
    audit_historical_rule_coverage,
)
from ssq_rule_registry import filter_pipeline_stats
from ssq_scoring import (
    get_omission,
    run_strategy_and_get_scores,
)
from ssq_selection_models import DuplexSelectionRequest
from ssq_training import (
    MODEL_TRAINING_PARAMS,
    train_prediction_models,
    validate_model_sets,
)

SCRIPT_DIR = _paths.SCRIPT_DIR
PROJECT_ROOT = _paths.PROJECT_ROOT
CSV_PATH = _paths.CSV_PATH
PARAMS_JSON_PATH = _paths.PARAMS_JSON_PATH
REPORT_DIR = _paths.REPORT_DIR
RUNTIME_PACKAGES = _report_output.RUNTIME_PACKAGES
PreparedHistory = _workflow_models.PreparedHistory
HistoricalEvaluation = _workflow_models.HistoricalEvaluation
CurrentSelection = _workflow_models.CurrentSelection


collect_runtime_versions = _report_output.collect_runtime_versions


def load_strategy_params(filepath=PARAMS_JSON_PATH):
    """Compatibility wrapper using the analyzer's default parameter path."""
    return _strategy_params.load_strategy_params(filepath)


def load_and_preprocess_data(filepath=CSV_PATH):
    """Compatibility wrapper using the analyzer's default history path."""
    return _history_preparation.load_and_preprocess_data(
        filepath,
        normalize_draw_frame,
        validate_draw_dates_not_future,
    )


def is_confirmation_input(value):
    """Compatibility wrapper for terminal confirmation parsing."""
    return _is_confirmation_input(value)


def get_user_input_with_timeout(timeout):
    """Compatibility wrapper for cross-platform timed input."""
    return _get_user_input_with_timeout(timeout)


def prepare_history():
    """Compatibility wrapper for preparing the default history file."""
    dependencies = _history_preparation.HistoryPreparationDependencies(
        load_data=load_and_preprocess_data,
        fingerprint_history=fingerprint_draw_frame,
        engineer_features=feature_engineer,
        infer_target_issue=infer_next_issue,
    )
    return _history_preparation.prepare_history(
        CSV_PATH,
        dependencies,
    )


def evaluate_history(history, options):
    """Compatibility wrapper for historical evaluation."""
    dependencies = _history_evaluation.HistoryEvaluationDependencies(
        load_params=load_strategy_params,
        audit_rules=audit_historical_rule_coverage,
        audit_hard_pipeline=audit_historical_hard_pipeline,
        run_backtest=run_backtest,
    )
    return _history_evaluation.evaluate_history(
        history,
        options,
        PARAMS_JSON_PATH,
        dependencies,
    )


def train_final_models(history):
    """Compatibility wrapper for final model training."""
    dependencies = _prediction_workflow.ModelTrainingDependencies(
        train_models=train_prediction_models,
        validate_models=validate_model_sets,
    )
    return _prediction_workflow.train_final_models(history, dependencies)


def select_current_issue(history, options, params, models):
    """Compatibility wrapper for current-issue candidate selection."""
    dependencies = _prediction_workflow.CurrentSelectionDependencies(
        score_balls=run_strategy_and_get_scores,
        derive_rejection_seed=rejection_seed_for_issue,
        build_rejection_set=make_rejection_set,
        get_omission=get_omission,
        generate_candidates=generate_candidates,
        collect_pipeline_stats=filter_pipeline_stats,
    )
    return _prediction_workflow.select_current_issue(
        history,
        options,
        params,
        models,
        dependencies,
    )


def display_passed_combinations(passed_combos, non_interactive):
    """Compatibility wrapper for candidate display policy."""
    return _display_passed_combinations(passed_combos, non_interactive)


def save_analysis_report(data):
    """Compatibility wrapper for analysis report persistence."""
    dependencies = _report_output.ReportOutputDependencies(
        build_report=build_analysis_report,
        write_text=atomic_write_text,
    )
    return _report_output.save_analysis_report(data, REPORT_DIR, dependencies)


def run_analysis(options):
    """Execute one complete analysis run and return the saved report path."""
    print('=' * 70)
    print('         双色球策略分析器 v7.0')
    print('=' * 70)

    print('\n[阶段 1/8] 正在加载和处理历史数据...')
    history = prepare_history()
    print('数据加载与特征工程完成。')

    print('\n[阶段 2/8] 正在执行历史规则审计与滚动回测...')
    evaluation = evaluate_history(history, options)

    print('\n[阶段 3/8] 正在使用全部历史数据，训练用于最终预测的模型...')
    models = train_final_models(history)

    print('\n[阶段 4/8] 正在为下一期号码进行机器学习评分...')
    current = select_current_issue(
        history,
        options,
        evaluation.loaded_params.values,
        models,
    )

    print('\n[阶段 6/8] 正在整理通过硬规则的候选组合...')
    display_passed_combinations(
        current.candidate_selection.passed_combos,
        options.non_interactive,
    )

    print('\n[阶段 7/8] 正在从最终组合中，生成高重合度的7红球大底...')
    best_7_reds = rank_duplex_candidates(DuplexSelectionRequest(
        passed_combos=current.candidate_selection.passed_combos,
        red_pool=current.candidate_selection.red_pool,
        red_scores=current.red_scores,
        context=current.rule_context,
    ))

    print('\n[阶段 8/8] 正在生成最终推荐报告...')
    generated_at = local_now()
    report_data = build_analysis_report_data(
        history,
        evaluation,
        current,
        options,
        best_7_reds,
        generated_at,
        collect_runtime_versions(),
        MODEL_TRAINING_PARAMS,
    )
    return save_analysis_report(report_data)


def main(argv=None):
    """Parse command-line arguments and run the analyzer workflow."""
    return run_analysis(parse_cli_options(argv))
