import unittest

import ssq_rule_registry as registry
import ssq_rules as rules


class RuleRegistryTests(unittest.TestCase):
    def test_rules_preserves_registry_compatibility_exports(self):
        self.assertIs(rules.RED_RULES, registry.RED_RULES)
        self.assertIs(rules.FILTER_NAMES, registry.FILTER_NAMES)
        self.assertIs(rules.HARD_FILTER_NAMES, registry.HARD_FILTER_NAMES)
        self.assertIs(rules.SOFT_FILTER_NAMES, registry.SOFT_FILTER_NAMES)
        self.assertIs(rules.passes_red_filters, registry.passes_red_filters)

    def test_prime_ratio_remains_an_ordinary_soft_rule(self):
        prime_rule = next(
            rule for rule in registry.RED_RULES
            if rule.name == 'prime_composite_ratio'
        )

        self.assertFalse(prime_rule.hard)
        self.assertEqual(prime_rule.score_weight, 0.08)
        self.assertIsNotNone(prime_rule.scorer)
        self.assertAlmostEqual(
            registry.COMBINATION_SIGNAL_WEIGHT
            + sum(rule.score_weight for rule in registry.RED_RULES),
            1.0,
        )


if __name__ == '__main__':
    unittest.main()
