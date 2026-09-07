import unittest

import ssq_rule_definitions as definitions
import ssq_rule_registry as registry


class RuleDefinitionTests(unittest.TestCase):
    def test_registry_preserves_definition_compatibility_exports(self):
        self.assertIs(registry.RED_RULES, definitions.RED_RULES)
        self.assertIs(registry.FILTER_NAMES, definitions.FILTER_NAMES)
        self.assertIs(registry.HARD_FILTER_NAMES, definitions.HARD_FILTER_NAMES)
        self.assertIs(registry.SOFT_FILTER_NAMES, definitions.SOFT_FILTER_NAMES)
        self.assertEqual(
            registry.COMBINATION_SIGNAL_WEIGHT,
            definitions.COMBINATION_SIGNAL_WEIGHT,
        )

    def test_rule_names_and_hard_soft_partitions_are_complete(self):
        names = tuple(rule.name for rule in definitions.RED_RULES)

        self.assertEqual(names, definitions.FILTER_NAMES)
        self.assertEqual(
            set(definitions.HARD_FILTER_NAMES)
            | set(definitions.SOFT_FILTER_NAMES),
            set(names),
        )
        self.assertFalse(
            set(definitions.HARD_FILTER_NAMES)
            & set(definitions.SOFT_FILTER_NAMES)
        )

    def test_prime_ratio_is_an_ordinary_soft_rule(self):
        rule = next(
            rule for rule in definitions.RED_RULES
            if rule.name == 'prime_composite_ratio'
        )

        self.assertFalse(rule.hard)
        self.assertEqual(rule.score_weight, 0.08)
        self.assertIsNotNone(rule.scorer)


if __name__ == '__main__':
    unittest.main()
