import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import ssq_backtest_reporting
import ssq_recommendation_reporting
import ssq_report_builder
import ssq_reporting
import ssq_rule_reporting


class ReportBuilderTests(unittest.TestCase):
    def test_reporting_preserves_report_builder_compatibility_export(self):
        self.assertIs(
            ssq_reporting.build_analysis_report,
            ssq_report_builder.build_analysis_report,
        )

    def test_build_report_assembles_specialized_sections_in_order(self):
        data = SimpleNamespace(
            latest_issue='2026103',
            target_issue=2026104,
            generated_at=datetime(2026, 9, 7, 12, 34, 56, tzinfo=timezone.utc),
            history_sha256='',
            runtime_versions={},
            model_features=(),
            model_training_params={},
        )
        with (
            patch.object(
                ssq_backtest_reporting,
                'format_backtest_report',
                return_value=['BACKTEST'],
            ),
            patch.object(
                ssq_rule_reporting,
                'format_rule_audit_report',
                return_value=['RULE AUDIT'],
            ),
            patch.object(
                ssq_recommendation_reporting,
                'format_recommendations_report',
                return_value=['RECOMMENDATIONS'],
            ),
        ):
            report = ssq_report_builder.build_analysis_report(data)

        self.assertLess(report.index('BACKTEST'), report.index('RULE AUDIT'))
        self.assertLess(report.index('RULE AUDIT'), report.index('RECOMMENDATIONS'))
        self.assertIn('Data_Basis_Issue: 2026103', report)
        self.assertIn('Prediction_Target_Issue: 2026104', report)
        self.assertIn('报告生成时间: 2026-09-07 12:34:56', report)


if __name__ == '__main__':
    unittest.main()
