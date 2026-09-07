import unittest

import ssq_combination_scoring as scoring
import ssq_ranking as ranking


class CombinationScoringTests(unittest.TestCase):
    def test_ranking_preserves_scoring_compatibility_exports(self):
        for name in (
            'build_rank_center_scores',
            'score_rank_center_preference',
            'build_combination_score_context',
            'score_combination',
            'score_red_combination',
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(ranking, name), getattr(scoring, name))

    def test_rank_signal_downweights_both_score_extremes(self):
        scores = {number: float(34 - number) for number in range(1, 34)}

        high = scoring.score_rank_center_preference((1, 2, 3, 4, 5, 6), scores)
        middle = scoring.score_rank_center_preference(
            (14, 15, 16, 17, 18, 19),
            scores,
        )
        low = scoring.score_rank_center_preference(
            (28, 29, 30, 31, 32, 33),
            scores,
        )

        self.assertGreater(middle, high)
        self.assertGreater(middle, low)


if __name__ == '__main__':
    unittest.main()
