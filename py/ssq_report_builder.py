"""Assembly of the complete analysis report."""

import ssq_backtest_reporting
import ssq_recommendation_reporting
import ssq_rule_reporting


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
    lines.extend(ssq_backtest_reporting.format_backtest_report(data))
    lines.extend(ssq_rule_reporting.format_rule_audit_report(data))
    lines.extend(ssq_recommendation_reporting.format_recommendations_report(data))
    lines.append("\n" + "=" * 60 + "\n报告结束。祝您好运！\n" + "=" * 60)
    return "\n".join(lines)
