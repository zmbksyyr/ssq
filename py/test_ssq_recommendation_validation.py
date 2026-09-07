import unittest

import ssq_ranking as ranking
import ssq_recommendation_validation as validation
from ssq_core import RED_BALLS
from ssq_rule_models import RecommendationRequest, RuleContext


class RecommendationValidationTests(unittest.TestCase):
    def test_ranking_preserves_validation_compatibility_export(self):
        self.assertIs(
            ranking.validate_recommendation_request,
            validation.validate_recommendation_request,
        )

    def test_request_materializes_generator_and_normalizes_scores(self):
        combos = ((1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 7))
        request = RecommendationRequest(
            passed_combos=(combo for combo in combos),
            red_scores={ball: ball for ball in RED_BALLS},
            context=RuleContext(),
            limit=1,
            max_shared=4,
        )

        actual_combos, scores, limit, max_shared = (
            validation.validate_recommendation_request(request)
        )

        self.assertEqual(actual_combos, combos)
        self.assertTrue(all(type(value) is float for value in scores.values()))
        self.assertEqual((limit, max_shared), (1, 4))
        self.assertEqual(tuple(request.passed_combos), ())


if __name__ == '__main__':
    unittest.main()
