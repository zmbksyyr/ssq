import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
from ssq_analysis_result import build_analysis_report_data
from ssq_config import StrategyConfig
from ssq_report_models import AnalysisReportData


class AnalysisResultTests(unittest.TestCase):
    def test_completed_stages_are_mapped_without_copying_domain_results(self):
        config = StrategyConfig()
        history = SimpleNamespace(
            latest_issue='2026103',
            target_issue=2026104,
            sha256='a' * 64,
            feature_columns=('feature_a', 'feature_b'),
        )
        loaded_params = SimpleNamespace(
            loaded_from_file=True,
            values={'weight': 1},
        )
        evaluation = SimpleNamespace(
            loaded_params=loaded_params,
            selected_backtest='selected-backtest',
            backtests={'mixed': 'mixed-backtest'},
            rule_coverage={'span': 0.99},
            hard_pipeline_coverage={'all': 0.95},
        )
        selection = object()
        current = SimpleNamespace(
            rejection_seed=123,
            pipeline_stats=[{'stage': 'span'}],
            candidate_selection=selection,
            recommended_blues=[7, 9],
        )
        options = SimpleNamespace(
            strategy_config=config,
            pool_mode='mixed',
            rule_audit_periods=200,
        )
        generated_at = datetime(2026, 9, 8, tzinfo=timezone.utc)
        runtime_versions = {'python': 'test'}
        training_params = MappingProxyType({'deterministic': True})

        result = build_analysis_report_data(
            history,
            evaluation,
            current,
            options,
            ['duplex'],
            generated_at,
            runtime_versions,
            training_params,
        )

        self.assertIsInstance(result, AnalysisReportData)
        self.assertEqual(result.latest_issue, '2026103')
        self.assertEqual(result.target_issue, 2026104)
        self.assertIs(result.config, config)
        self.assertIs(result.params, loaded_params.values)
        self.assertIs(result.selection, selection)
        self.assertIs(result.backtests, evaluation.backtests)
        self.assertIs(result.runtime_versions, runtime_versions)
        self.assertIs(result.model_training_params, training_params)
        self.assertEqual(result.rank_band_widths['middle'], 9)
        self.assertEqual(result.rank_band_labels['middle'], '中段(13-21)')


if __name__ == '__main__':
    unittest.main()
