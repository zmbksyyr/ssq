import unittest

import ssq_core as core
import ssq_parsing as parsing
from ssq_domain import RED_BALLS


class ParsingTests(unittest.TestCase):
    def test_core_preserves_parsing_compatibility_exports(self):
        for name in (
            'parse_integer',
            'parse_issue',
            'parse_red_balls',
            'parse_blue_ball',
            'parse_blue_balls',
            'validate_ball_scores',
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(parsing, name))

    def test_score_validation_normalizes_complete_domain(self):
        scores = parsing.validate_ball_scores(
            {ball: ball for ball in RED_BALLS},
            RED_BALLS,
            '红球',
        )

        self.assertEqual(set(scores), set(RED_BALLS))
        self.assertTrue(all(type(value) is float for value in scores.values()))

    def test_score_validation_rejects_missing_and_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, '评分键不完整'):
            parsing.validate_ball_scores({1: 0.5}, RED_BALLS, '红球')
        with self.assertRaisesRegex(ValueError, '有限数值'):
            parsing.validate_ball_scores(
                {ball: float('nan') for ball in RED_BALLS},
                RED_BALLS,
                '红球',
            )


if __name__ == '__main__':
    unittest.main()
