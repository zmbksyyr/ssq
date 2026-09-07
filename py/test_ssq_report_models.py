import unittest

import ssq_report_models as models
import ssq_reporting as reporting


class ReportModelTests(unittest.TestCase):
    def test_reporting_preserves_data_model_compatibility_export(self):
        self.assertIs(reporting.AnalysisReportData, models.AnalysisReportData)

    def test_optional_metadata_defaults_are_not_shared(self):
        required = {
            'latest_issue': '2026103',
            'target_issue': 2026104,
            'generated_at': object(),
            'params_loaded': False,
            'params': {},
            'config': object(),
            'rejection_seed': 1,
            'backtest': object(),
            'backtests': {},
            'pool_mode': 'mixed',
            'rank_band_widths': {},
            'rank_band_labels': {},
            'pipeline_stats': [],
            'rule_coverage': {},
            'hard_pipeline_coverage': {},
            'rule_audit_periods': 0,
            'selection': object(),
            'recommended_blues': [],
            'best_7_reds': [],
        }
        first = models.AnalysisReportData(**required)
        second = models.AnalysisReportData(**required)

        self.assertIsNot(first.runtime_versions, second.runtime_versions)
        self.assertIsNot(first.model_training_params, second.model_training_params)


if __name__ == '__main__':
    unittest.main()
