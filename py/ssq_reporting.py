"""Pure text formatting for complete analysis reports."""

import ssq_backtest_reporting as _backtest_reporting
import ssq_recommendation_reporting as _recommendation_reporting
import ssq_report_models as _report_models
import ssq_rule_reporting as _rule_reporting

AnalysisReportData = _report_models.AnalysisReportData
PRIZE_DISPLAY_ORDER = _backtest_reporting.PRIZE_DISPLAY_ORDER
format_strategy_parameters = _backtest_reporting.format_strategy_parameters
format_backtest_metrics = _backtest_reporting.format_backtest_metrics
format_pool_comparison = _backtest_reporting.format_pool_comparison
format_window_stability = _backtest_reporting.format_window_stability
format_rank_band_distribution = _backtest_reporting.format_rank_band_distribution
format_prize_counts = _backtest_reporting.format_prize_counts
format_backtest_report = _backtest_reporting.format_backtest_report
format_audit_window = _rule_reporting.format_audit_window
format_rule_audit_report = _rule_reporting.format_rule_audit_report
format_recommendations_report = (
    _recommendation_reporting.format_recommendations_report
)


def build_analysis_report(data):
    lines = [
        "=" * 60,
        "          双色球策略分析与推荐报告 (高级过滤版)",
        "=" * 60,
        "\n--- 0. 报告元数据 ---",
        f"Data_Basis_Issue: {data.latest_issue}",
        f"Prediction_Target_Issue: {data.target_issue}",
        f"报告生成时间: {data.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if data.history_sha256:
        lines.append(f'Data_History_SHA256: {data.history_sha256}')
    for name, version in data.runtime_versions.items():
        lines.append(f'Runtime_{name}: {version}')
    if data.model_features:
        lines.append(f"Model_Features: {','.join(data.model_features)}")
    for name, value in data.model_training_params.items():
        lines.append(f'Model_LightGBM_{name}: {value}')
    lines.extend(format_backtest_report(data))
    lines.extend(format_rule_audit_report(data))
    lines.extend(format_recommendations_report(data))
    lines.append("\n" + "=" * 60 + "\n报告结束。祝您好运！\n" + "=" * 60)
    return "\n".join(lines)
