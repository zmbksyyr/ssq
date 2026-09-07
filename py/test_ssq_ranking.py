import unittest
from unittest.mock import patch

import ssq_ranking as ranking
import ssq_rules as rules
import ssq_selection as selection
from ssq_rule_models import RuleContext, RuleDefinition


class RankingTests(unittest.TestCase):
    def test_selection_uses_the_dedicated_ranking_module(self):
        self.assertIs(
            selection.build_combination_score_context,
            ranking.build_combination_score_context,
        )
        self.assertIs(selection.score_combination, ranking.score_combination)
        self.assertIs(
            selection.select_recommendation_portfolio,
            ranking.select_recommendation_portfolio,
        )

    def test_rules_compatibility_wrapper_uses_its_active_registry(self):
        scores = {ball: float(ball) for ball in range(1, 34)}
        combo = (3, 8, 14, 21, 27, 32)
        context = RuleContext(omission_values={1: 7})
        context_rule = RuleDefinition(
            'context_rule',
            False,
            lambda _combo, _context: True,
            0.25,
            lambda _combo, actual_context: actual_context.omission_values[1] / 7,
        )

        with patch.object(rules, 'RED_RULES', (context_rule,)):
            actual = rules.score_red_combination(combo, scores, context=context)

        signal = ranking.score_rank_center_preference(combo, scores)
        self.assertAlmostEqual(actual, 0.50 * signal + 0.25)


if __name__ == '__main__':
    unittest.main()
