import unittest

import ssq_candidates as candidates
import ssq_selection as selection


class CandidateModuleTests(unittest.TestCase):
    def test_selection_preserves_candidate_compatibility_exports(self):
        self.assertIs(selection.build_red_pool, candidates.build_red_pool)
        self.assertIs(
            selection.count_actual_reds_by_rank_band,
            candidates.count_actual_reds_by_rank_band,
        )
        self.assertIs(selection.generate_candidates, candidates.generate_candidates)
        self.assertIs(
            selection.generate_red_candidates,
            candidates.generate_red_candidates,
        )
        self.assertIs(
            selection.validate_candidate_generation_request,
            candidates.validate_candidate_generation_request,
        )

    def test_default_mixed_pool_keeps_high_middle_low_counts(self):
        scores = {ball: float(34 - ball) for ball in range(1, 34)}

        self.assertEqual(
            candidates.build_red_pool(scores),
            [1, 2, 3, 4, *range(13, 22), 30, 31, 32, 33],
        )


if __name__ == '__main__':
    unittest.main()
