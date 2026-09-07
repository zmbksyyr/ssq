import ast
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import ssq_ball_scoring as ball_scoring
from ssq_domain import BLUE_BALLS, RED_BALLS


class BallScoringTests(unittest.TestCase):
    def test_fusion_uses_injected_calculation_boundaries(self):
        params = {
            'decay_factor': 0.9,
            'weight_freq': 0.4,
            'weight_omission': 0.5,
            'weight_ml': 0.1,
            'weight_blue_freq': 0.6,
            'weight_blue_ml': 0.4,
        }
        history = pd.DataFrame({
            '红球': [[1, 2, 3, 4, 5, 6]],
            '蓝球': [1],
            'feature': [0.0],
        })
        red_models = {ball: object() for ball in RED_BALLS}
        blue_models = {ball: object() for ball in BLUE_BALLS}
        dependencies = ball_scoring.BallScoringDependencies(
            validate_params=Mock(return_value=params),
            validate_models=Mock(),
            validate_features=Mock(return_value=('feature',)),
            predict_probability=Mock(return_value=np.array([1.0])),
            weighted_frequency=Mock(side_effect=(
                pd.Series({ball: 1.0 for ball in RED_BALLS}),
                pd.Series({ball: 1.0 for ball in BLUE_BALLS}),
            )),
            omission=Mock(return_value={ball: 1 for ball in RED_BALLS}),
            adjust_red_scores=Mock(side_effect=lambda scores, *_: scores),
            validate_scores=Mock(side_effect=lambda scores, *_: scores),
        )

        red_scores, blue_scores = ball_scoring.run_strategy_and_get_scores(
            history,
            params,
            red_models,
            blue_models,
            ('feature',),
            dependencies,
        )

        self.assertTrue(all(score == 1.0 for score in red_scores.values()))
        self.assertTrue(all(score == 1.0 for score in blue_scores.values()))
        dependencies.validate_models.assert_called_once_with(
            red_models,
            blue_models,
        )
        self.assertEqual(dependencies.weighted_frequency.call_count, 2)

    def test_production_workflows_do_not_import_scoring_facade(self):
        filenames = (
            'ssq_backtesting.py',
            'ssq_backtest_preparation.py',
            'ssq_prediction_workflow.py',
            'ssq_workflow.py',
        )
        for filename in filenames:
            tree = ast.parse(
                Path(__file__).with_name(filename).read_text(encoding='utf-8')
            )
            imports = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            with self.subTest(filename=filename):
                self.assertNotIn('ssq_scoring', imports)


if __name__ == '__main__':
    unittest.main()
