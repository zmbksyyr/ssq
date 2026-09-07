import unittest

import ssq_rule_functions as functions
import ssq_rule_registry as registry


class RuleFunctionTests(unittest.TestCase):
    def test_registry_preserves_rule_function_compatibility_exports(self):
        names = (
            'is_prime',
            'calculate_ac_value',
            'filter_prime_composite_ratio',
            'filter_big_small_ratio',
            'score_prime_balance',
            'score_big_small_balance',
        )
        for name in names:
            with self.subTest(name=name):
                self.assertIs(getattr(registry, name), getattr(functions, name))

    def test_prime_ratio_is_evaluated_like_other_balance_ratios(self):
        self.assertFalse(functions.filter_prime_composite_ratio((1, 4, 6, 8, 9, 10)))
        self.assertTrue(functions.filter_prime_composite_ratio((2, 3, 4, 6, 8, 9)))
        self.assertEqual(functions.score_prime_balance((2, 3, 5, 6, 8, 9)), 1.0)


if __name__ == '__main__':
    unittest.main()
