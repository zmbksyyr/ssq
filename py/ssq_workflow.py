"""Application workflow for loading data, running analysis, and saving reports."""

import os
import platform
from importlib.metadata import version

import ssq_history_evaluation as _history_evaluation
import ssq_history_preparation as _history_preparation
import ssq_prediction_workflow as _prediction_workflow
import ssq_strategy_params as _strategy_params
import ssq_workflow_models as _workflow_models
from ssq_anti_crowding import make_rejection_set, rejection_seed_for_issue
from ssq_backtesting import run_backtest
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
from ssq_core import (
    atomic_write_text,
    infer_next_issue,
    local_now,
)
from ssq_draw_data import (
    fingerprint_draw_frame,
    normalize_draw_frame,
    validate_draw_dates_not_future,
)
from ssq_duplex import rank_duplex_candidates
from ssq_features import feature_engineer
from ssq_rank_bands import build_rank_band_labels, build_rank_band_widths
from ssq_report_builder import build_analysis_report
from ssq_report_models import AnalysisReportData
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(PROJECT_ROOT, 'shuangseqiu.csv')
PARAMS_JSON_PATH = os.path.join(PROJECT_ROOT, 'best_params.json')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')
RUNTIME_PACKAGES = ('numpy', 'pandas', 'lightgbm', 'scikit-learn')
PreparedHistory = _workflow_models.PreparedHistory
HistoricalEvaluation = _workflow_models.HistoricalEvaluation
CurrentSelection = _workflow_models.CurrentSelection


def collect_runtime_versions():
    return {
        'python': platform.python_version(),
        **{package: version(package) for package in RUNTIME_PACKAGES},
    }


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
    """Build, print, and atomically persist an analysis report."""
    report = build_analysis_report(data)
    print('\n\n' + report)
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        timestamp = data.generated_at.strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(REPORT_DIR, f'ssq_analysis_output_{timestamp}.txt')
        atomic_write_text(filepath, report)
    except OSError as exc:
        raise SystemExit(f'\n\n写入报告文件失败: {exc}') from exc
    print(f'\n\n报告已成功保存到文件: {filepath}')
    return filepath


def run_analysis(options):
    """Execute one complete analysis run and return the saved report path."""
    config = options.strategy_config

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
    report_data = AnalysisReportData(
        latest_issue=history.latest_issue,
        target_issue=history.target_issue,
        generated_at=generated_at,
        params_loaded=evaluation.loaded_params.loaded_from_file,
        params=evaluation.loaded_params.values,
        config=config,
        rejection_seed=current.rejection_seed,
        backtest=evaluation.selected_backtest,
        backtests=evaluation.backtests,
        pool_mode=options.pool_mode,
        rank_band_widths=build_rank_band_widths(config),
        rank_band_labels=build_rank_band_labels(config),
        pipeline_stats=current.pipeline_stats,
        rule_coverage=evaluation.rule_coverage,
        hard_pipeline_coverage=evaluation.hard_pipeline_coverage,
        rule_audit_periods=options.rule_audit_periods,
        selection=current.candidate_selection,
        recommended_blues=current.recommended_blues,
        best_7_reds=best_7_reds,
        runtime_versions=collect_runtime_versions(),
        history_sha256=history.sha256,
        model_features=history.feature_columns,
        model_training_params=MODEL_TRAINING_PARAMS,
    )
    return save_analysis_report(report_data)


def main(argv=None):
    """Parse command-line arguments and run the analyzer workflow."""
    return run_analysis(parse_cli_options(argv))
