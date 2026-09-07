import unittest

import ssq_ranking as ranking
import ssq_recommendation_portfolio as portfolio
from ssq_core import RED_BALLS
from ssq_rule_models import RecommendationRequest, RuleContext


class RecommendationPortfolioTests(unittest.TestCase):
    def test_ranking_preserves_portfolio_compatibility_export(self):
        self.assertIs(
            ranking.select_recommendation_portfolio,
            portfolio.select_recommendation_portfolio,
        )

    def test_selection_relaxes_overlap_only_to_fill_requested_count(self):
        combos = (
            (1, 2, 3, 4, 5, 6),
            (1, 2, 3, 4, 5, 7),
            (1, 2, 3, 4, 5, 8),
        )
        request = RecommendationRequest(
            passed_combos=combos,
            red_scores={ball: float(ball) for ball in RED_BALLS},
            context=RuleContext(),
            limit=3,
            max_shared=4,
        )

        selected = portfolio.select_recommendation_portfolio(
            request,
            rule_definitions=(),
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual(set(selected), set(combos))
        self.assertEqual(len(set(selected[0]) & set(selected[1])), 5)

    def test_zero_limit_avoids_scoring(self):
        request = RecommendationRequest(
            passed_combos=((1, 2, 3, 4, 5, 6),),
            red_scores={ball: float(ball) for ball in RED_BALLS},
            context=RuleContext(),
            limit=0,
            max_shared=4,
        )

        self.assertEqual(portfolio.select_recommendation_portfolio(request), [])


if __name__ == '__main__':
    unittest.main()
