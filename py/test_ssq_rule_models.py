import unittest

import ssq_rule_models as models
import ssq_rules as rules


class RuleModelTests(unittest.TestCase):
    def test_rules_preserves_model_compatibility_exports(self):
        self.assertIs(rules.RuleContext, models.RuleContext)
        self.assertIs(rules.RuleDefinition, models.RuleDefinition)
        self.assertIs(
            rules.CombinationScoreContext,
            models.CombinationScoreContext,
        )
        self.assertIs(rules.RecommendationRequest, models.RecommendationRequest)

    def test_rule_context_default_collections_are_not_shared(self):
        first = models.RuleContext()
        second = models.RuleContext()

        self.assertIsNot(first.omission_values, second.omission_values)
        self.assertEqual(first.recent_draws, ())
        self.assertIsNone(first.last_draw)
        self.assertIsNone(first.previous_draw)


if __name__ == '__main__':
    unittest.main()
