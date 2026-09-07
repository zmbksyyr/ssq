import unittest

import ssq_rule_registry as registry
import ssq_rule_validation as validation
from ssq_rule_models import RuleDefinition


class RuleValidationTests(unittest.TestCase):
    def test_registry_preserves_validation_compatibility_exports(self):
        self.assertIs(registry.validate_signal_weight, validation.validate_signal_weight)
        self.assertIs(registry.validate_rule_names, validation.validate_rule_names)
        self.assertIs(
            registry.validate_rule_definition,
            validation.validate_rule_definition,
        )

    def test_validation_rejects_unnormalized_score_budget(self):
        rule = RuleDefinition(
            'soft',
            False,
            lambda _combo, _context: True,
            0.2,
            lambda _combo, _context: 1.0,
        )

        with self.assertRaisesRegex(ValueError, '组合评分总权重必须为 1'):
            validation.validate_rule_registry((rule,), 0.5)

    def test_validation_accepts_generator_input_once(self):
        actual = validation.validate_rule_registry(
            (rule for rule in registry.RED_RULES),
            registry.COMBINATION_SIGNAL_WEIGHT,
        )

        self.assertEqual(actual, registry.RED_RULES)


if __name__ == '__main__':
    unittest.main()
