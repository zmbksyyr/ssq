"""Deterministic assembly of complete analysis-report data."""

from ssq_rank_bands import build_rank_band_labels, build_rank_band_widths
from ssq_report_models import AnalysisReportData


def build_analysis_report_data(
    history,
    evaluation,
    current,
    options,
    best_7_reds,
    generated_at,
    runtime_versions,
    model_training_params,
):
    """Map completed workflow stages into the report data contract."""
    config = options.strategy_config
    return AnalysisReportData(
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
        runtime_versions=runtime_versions,
        history_sha256=history.sha256,
        model_features=history.feature_columns,
        model_training_params=model_training_params,
    )
