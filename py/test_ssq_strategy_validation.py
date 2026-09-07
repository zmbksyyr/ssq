import unittest

import ssq_config as config
import ssq_strategy_validation as validation


class StrategyValidationTests(unittest.TestCase):
    def test_config_preserves_validation_compatibility_exports(self):
        for name in (
            'normalize_integer_param',
            'normalize_float_param',
            'validate_weight_group',
            'validate_param_ranges',
            'validate_strategy_params',
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(config, name), getattr(validation, name))

    def test_default_scoring_weights_preserve_strategy(self):
        params = validation.validate_strategy_params({})

        self.assertEqual(
            [params[name] for name in (
                'weight_freq', 'weight_omission', 'weight_ml',
            )],
            [0.4, 0.5, 0.1],
        )
        self.assertEqual(
            [params[name] for name in ('weight_blue_freq', 'weight_blue_ml')],
            [0.6, 0.4],
        )


if __name__ == '__main__':
    unittest.main()
