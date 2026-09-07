import io
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import ssq_prediction_workflow as prediction
from ssq_config import AnalyzerOptions
from ssq_selection_models import RedCandidateSelection
from ssq_workflow_models import PreparedHistory


class PredictionWorkflowTests(unittest.TestCase):
    def test_final_training_uses_history_after_feature_warmup(self):
        frame = pd.DataFrame({'期号': range(30)})
        history = PreparedHistory(frame, ('feature',), '29', 30, 'a' * 64)
        models = ('red models', 'blue models')
        train = Mock(return_value=models)
        validate = Mock()

        result = prediction.train_final_models(
            history,
            prediction.ModelTrainingDependencies(train, validate),
        )

        self.assertEqual(result, models)
        training_frame = train.call_args.args[0]
        self.assertEqual(training_frame['期号'].tolist(), list(range(5, 30)))
        train.assert_called_once_with(
            training_frame,
            ('feature',),
            show_progress=True,
        )
        validate.assert_called_once_with(*models)

    def test_selection_preserves_seed_blue_ranking_and_rule_context(self):
        frame = pd.DataFrame({
            '红球': [list(range(index, index + 6)) for index in range(1, 11)],
        })
        history = PreparedHistory(frame, ('feature',), '2026001', 2026002, '')
        options = AnalyzerOptions(rejection_size=500_000, seed=42)
        red_scores = {ball: float(ball) for ball in range(1, 34)}
        blue_scores = {ball: float(ball) for ball in range(1, 17)}
        selection = RedCandidateSelection(
            tuple(range(1, 18)),
            ((1, 2, 3, 4, 5, 6),),
            ((1, 2, 3, 4, 5, 6),),
            ((1, 2, 3, 4, 5, 6),),
        )
        build_rejection_set = Mock(return_value=set())
        generate = Mock(return_value=selection)
        dependencies = prediction.CurrentSelectionDependencies(
            score_balls=Mock(return_value=(red_scores, blue_scores)),
            derive_rejection_seed=Mock(return_value=123),
            build_rejection_set=build_rejection_set,
            get_omission=Mock(return_value={1: 5}),
            generate_candidates=generate,
            collect_pipeline_stats=Mock(return_value=[{'rule': 'sum_range'}]),
        )

        with patch('sys.stdout', new_callable=io.StringIO):
            result = prediction.select_current_issue(
                history,
                options,
                {'weight': 1},
                ('red models', 'blue models'),
                dependencies,
            )

        self.assertEqual(result.recommended_blues, [16, 15, 14, 13, 12, 11, 10])
        self.assertEqual(result.rejection_seed, 123)
        self.assertEqual(result.pipeline_stats, [{'rule': 'sum_range'}])
        dependencies.derive_rejection_seed.assert_called_once_with(42, 2026002)
        self.assertEqual(build_rejection_set.call_args.args[0], 500_000)
        request = generate.call_args.args[0]
        self.assertEqual(request.mode, 'mixed')
        self.assertTrue(request.show_progress)
        self.assertEqual(request.context.omission_values, {1: 5})
        self.assertEqual(request.context.last_draw, set(range(10, 16)))
        self.assertEqual(request.context.previous_draw, set(range(9, 15)))


if __name__ == '__main__':
    unittest.main()
