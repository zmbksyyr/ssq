import unittest

import pandas as pd
import ssq_score_adjustments as adjustments
import ssq_scoring as scoring
from ssq_domain import RED_BALLS


class ScoreAdjustmentTests(unittest.TestCase):
    def test_scoring_preserves_adjustment_compatibility_export(self):
        self.assertIs(
            scoring.apply_red_score_adjustments,
            adjustments.apply_red_score_adjustments,
        )

    def test_hot_cold_and_repeat_multipliers_compose(self):
        history = pd.DataFrame({'红球': [
            [1, 2, 3, 4, 5, 6],
            [1, 7, 8, 9, 10, 11],
        ]})
        scores = {ball: 1.0 for ball in RED_BALLS}
        params = {
            'hot_lookback': 2,
            'hot_threshold': 2,
            'hot_bonus': 2.0,
            'cold_lookback': 1,
            'cold_bonus': 3.0,
            'repeat_bonus': 5.0,
        }

        result = adjustments.apply_red_score_adjustments(
            scores,
            history,
            params,
        )

        self.assertEqual(result[1], 10.0)
        self.assertEqual(result[2], 3.0)
        self.assertEqual(result[7], 5.0)
        self.assertEqual(result[33], 3.0)
        self.assertEqual(scores[1], 1.0)


if __name__ == '__main__':
    unittest.main()
