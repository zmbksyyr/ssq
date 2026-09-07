import unittest

import ssq_candidates as candidates
import ssq_red_pool as red_pool
from ssq_config import DEFAULT_STRATEGY_CONFIG


class RedPoolTests(unittest.TestCase):
    def test_candidates_preserves_red_pool_compatibility_exports(self):
        self.assertIs(candidates.build_red_pool, red_pool.build_red_pool)
        self.assertIs(
            candidates.count_actual_reds_by_rank_band,
            red_pool.count_actual_reds_by_rank_band,
        )

    def test_equal_scores_use_ball_number_as_stable_rank_tiebreaker(self):
        scores = {ball: 1.0 for ball in range(1, 34)}

        result = red_pool.build_red_pool(scores)

        self.assertEqual(
            result,
            [1, 2, 3, 4, *range(13, 22), 30, 31, 32, 33],
        )
        self.assertEqual(len(result), DEFAULT_STRATEGY_CONFIG.pool_size_red)

    def test_actual_numbers_are_counted_by_score_rank_band(self):
        scores = {ball: float(34 - ball) for ball in range(1, 34)}

        counts = red_pool.count_actual_reds_by_rank_band(
            scores,
            {1, 13, 21, 30, 33, 25},
        )

        self.assertEqual(
            counts,
            {'high': 1, 'middle': 2, 'low': 2, 'other': 1},
        )


if __name__ == '__main__':
    unittest.main()
