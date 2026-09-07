import unittest

import ssq_rule_functions as functions
import ssq_rule_registry as registry
import ssq_rule_scoring as scoring


class RuleScoringTests(unittest.TestCase):
    def test_legacy_and_registry_exports_share_score_functions(self):
        for name in (
            'score_zone_balance',
            'score_odd_even_balance',
            'score_prime_balance',
            'score_big_small_balance',
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(functions, name), getattr(scoring, name))
                self.assertIs(getattr(registry, name), getattr(scoring, name))

    def test_prime_balance_remains_a_normal_soft_score(self):
        self.assertEqual(scoring.score_prime_balance((2, 3, 5, 6, 8, 9)), 1.0)
        prime_rule = next(
            rule for rule in registry.RED_RULES
            if rule.name == 'prime_composite_ratio'
        )
        self.assertFalse(prime_rule.hard)
        self.assertEqual(prime_rule.score_weight, 0.08)


if __name__ == '__main__':
    unittest.main()
