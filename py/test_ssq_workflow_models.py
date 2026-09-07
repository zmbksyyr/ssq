import unittest

import pandas as pd
import ssq_workflow as workflow
import ssq_workflow_models as models
from ssq_config import LoadedStrategyParams


class WorkflowModelTests(unittest.TestCase):
    def test_workflow_preserves_model_compatibility_exports(self):
        for name in ('PreparedHistory', 'HistoricalEvaluation', 'CurrentSelection'):
            with self.subTest(name=name):
                self.assertIs(getattr(workflow, name), getattr(models, name))

    def test_prepared_history_retains_run_identity(self):
        frame = pd.DataFrame({'期号': [2026001]})

        history = models.PreparedHistory(
            frame=frame,
            feature_columns=('red_sum',),
            latest_issue='2026001',
            target_issue=2026002,
            sha256='a' * 64,
        )
        evaluation = models.HistoricalEvaluation(
            loaded_params=LoadedStrategyParams({'weight': 1}, True),
            rule_coverage={},
            hard_pipeline_coverage={},
            backtests={'mixed': 'result'},
            selected_backtest='result',
        )

        self.assertIs(history.frame, frame)
        self.assertEqual(history.target_issue, 2026002)
        self.assertEqual(evaluation.selected_backtest, 'result')


if __name__ == '__main__':
    unittest.main()
