import ast
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import ssq_model_training as model_training
from ssq_domain import BLUE_BALLS, RED_BALLS
from ssq_model_contracts import MODEL_TRAINING_PARAMS, BallModelSpec


class ModelTrainingTests(unittest.TestCase):
    def test_model_fit_uses_injected_classifier(self):
        frame = pd.DataFrame({
            'feature': [1.0, 2.0, 3.0],
            'outcome': [[1], [2], [1]],
        })
        model = Mock()
        classifier = Mock(return_value=model)
        dependencies = model_training.ModelFitDependencies(
            classifier_factory=classifier,
            progress_factory=Mock(),
        )
        spec = BallModelSpec(
            candidates=(1,),
            outcome_column='outcome',
            contains_candidate=lambda draw, ball: ball in draw,
        )

        models = model_training.train_models_for_spec(
            frame,
            ('feature',),
            spec,
            dependencies,
        )

        self.assertIs(models[1], model)
        classifier.assert_called_once_with(**MODEL_TRAINING_PARAMS)
        fitted_features, fitted_target = model.fit.call_args.args
        self.assertEqual(fitted_features['feature'].tolist(), [1.0, 2.0])
        self.assertEqual(fitted_target.tolist(), [0.0, 1.0])

    def test_prediction_training_builds_complete_red_and_blue_specs(self):
        validate = Mock(return_value=('feature',))
        train_spec = Mock(side_effect=({'red': 'models'}, {'blue': 'models'}))
        dependencies = model_training.PredictionTrainingDependencies(
            validate_features=validate,
            train_spec=train_spec,
        )
        frame = pd.DataFrame()

        result = model_training.train_prediction_models(
            frame,
            ('feature',),
            show_progress=True,
            dependencies=dependencies,
        )

        self.assertEqual(result, ({'red': 'models'}, {'blue': 'models'}))
        red_spec = train_spec.call_args_list[0].args[2]
        blue_spec = train_spec.call_args_list[1].args[2]
        self.assertEqual(red_spec.candidates, RED_BALLS)
        self.assertEqual(blue_spec.candidates, BLUE_BALLS)
        self.assertEqual(red_spec.description, '训练红球模型')
        self.assertEqual(blue_spec.description, '训练蓝球模型')

    def test_production_workflows_do_not_import_training_facade(self):
        filenames = (
            'ssq_backtest_preparation.py',
            'ssq_backtesting.py',
            'ssq_ball_scoring.py',
            'ssq_prediction_workflow.py',
            'ssq_scoring.py',
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
                self.assertNotIn('ssq_training', imports)


if __name__ == '__main__':
    unittest.main()
