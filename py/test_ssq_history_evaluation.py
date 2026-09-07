import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import ssq_history_evaluation as evaluation
from ssq_config import AnalyzerOptions, LoadedStrategyParams
from ssq_workflow_models import PreparedHistory


class HistoryEvaluationTests(unittest.TestCase):
    def test_evaluation_maps_audits_and_pool_result(self):
        history = PreparedHistory(
            frame=pd.DataFrame(),
            feature_columns=('feature',),
            latest_issue='2026001',
            target_issue=2026002,
            sha256='a' * 64,
        )
        options = AnalyzerOptions(backtest_periods=3, rejection_size=0)
        loaded = LoadedStrategyParams({'weight': 1}, True)
        run_backtest = Mock(return_value={'mixed': 'result'})
        dependencies = evaluation.HistoryEvaluationDependencies(
            load_params=Mock(return_value=loaded),
            audit_rules=Mock(return_value={'rule': 'coverage'}),
            audit_hard_pipeline=Mock(return_value={'hard': 'coverage'}),
            run_backtest=run_backtest,
        )

        result = evaluation.evaluate_history(
            history,
            options,
            'params.json',
            dependencies,
        )

        self.assertIs(result.loaded_params, loaded)
        self.assertEqual(result.rule_coverage, {'rule': 'coverage'})
        self.assertEqual(result.hard_pipeline_coverage, {'hard': 'coverage'})
        self.assertEqual(result.selected_backtest, 'result')
        request = run_backtest.call_args.args[1]
        self.assertEqual(request.feature_columns, ('feature',))
        self.assertEqual(request.num_periods, 3)
        self.assertEqual(request.config, options.strategy_config)

    def test_invalid_parameter_file_becomes_workflow_exit(self):
        dependencies = evaluation.HistoryEvaluationDependencies(
            load_params=Mock(side_effect=ValueError('invalid params')),
            audit_rules=Mock(),
            audit_hard_pipeline=Mock(),
            run_backtest=Mock(),
        )

        with (
            patch('sys.stdout', new_callable=io.StringIO),
            self.assertRaisesRegex(SystemExit, '错误: invalid params'),
        ):
            evaluation.evaluate_history(
                SimpleNamespace(frame=pd.DataFrame(), feature_columns=()),
                AnalyzerOptions(rejection_size=0),
                'params.json',
                dependencies,
            )

        dependencies.audit_rules.assert_not_called()


if __name__ == '__main__':
    unittest.main()
