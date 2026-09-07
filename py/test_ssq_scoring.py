import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import ssq_modeling as modeling
import ssq_scoring as scoring
from ssq_config import DEFAULT_PARAMS
from ssq_core import BLUE_BALLS, RED_BALLS
from ssq_features import FEATURE_COLUMNS


class ScoringModuleTests(unittest.TestCase):
    def test_modeling_preserves_scoring_compatibility_exports(self):
        self.assertIs(modeling.get_omission, scoring.get_omission)
        self.assertIs(
            modeling.get_weighted_frequency,
            scoring.get_weighted_frequency,
        )
        self.assertIs(
            modeling.apply_red_score_adjustments,
            scoring.apply_red_score_adjustments,
        )
        self.assertIs(
            modeling.run_strategy_and_get_scores,
            scoring.run_strategy_and_get_scores,
        )

    def test_signal_fusion_preserves_declared_red_and_blue_weights(self):
        history = pd.DataFrame({
            '红球': [[1, 2, 3, 4, 5, 6]],
            '蓝球': [1],
            **{column: [0.0] for column in FEATURE_COLUMNS},
        })
        red_frequency = pd.Series({ball: 1.0 for ball in RED_BALLS})
        blue_frequency = pd.Series({ball: 1.0 for ball in BLUE_BALLS})
        red_models = {ball: object() for ball in RED_BALLS}
        blue_models = {ball: object() for ball in BLUE_BALLS}

        with (
            patch.object(scoring, 'validate_model_sets'),
            patch.object(
                scoring,
                'get_weighted_frequency',
                side_effect=(red_frequency, blue_frequency),
            ),
            patch.object(
                scoring,
                'get_omission',
                return_value={ball: 1 for ball in RED_BALLS},
            ),
            patch.object(
                scoring,
                'predict_positive_probability',
                return_value=np.array([1.0]),
            ),
            patch.object(
                scoring,
                'apply_red_score_adjustments',
                side_effect=lambda scores, _history, _params: scores,
            ),
        ):
            red_scores, blue_scores = scoring.run_strategy_and_get_scores(
                history,
                DEFAULT_PARAMS,
                red_models,
                blue_models,
                FEATURE_COLUMNS,
            )

        self.assertTrue(all(score == 1.0 for score in red_scores.values()))
        self.assertTrue(all(score == 1.0 for score in blue_scores.values()))
        self.assertEqual(
            (
                DEFAULT_PARAMS['weight_freq'],
                DEFAULT_PARAMS['weight_omission'],
                DEFAULT_PARAMS['weight_ml'],
            ),
            (0.4, 0.5, 0.1),
        )
        self.assertEqual(
            (
                DEFAULT_PARAMS['weight_blue_freq'],
                DEFAULT_PARAMS['weight_blue_ml'],
            ),
            (0.6, 0.4),
        )


if __name__ == '__main__':
    unittest.main()
