import unittest

import ssq_config as config
import ssq_config_models as models


class ConfigModelTests(unittest.TestCase):
    def test_config_preserves_model_compatibility_exports(self):
        self.assertIs(config.StrategyConfig, models.StrategyConfig)
        self.assertIs(config.AnalyzerOptions, models.AnalyzerOptions)
        self.assertIs(config.LoadedStrategyParams, models.LoadedStrategyParams)
        self.assertIs(config.DEFAULT_STRATEGY_CONFIG, models.DEFAULT_STRATEGY_CONFIG)

    def test_default_selection_shape_preserves_strategy(self):
        strategy = models.DEFAULT_STRATEGY_CONFIG

        self.assertEqual(strategy.pool_size_red, 17)
        self.assertEqual(strategy.high_count, 4)
        self.assertEqual(
            strategy.pool_size_red - strategy.high_count - strategy.low_count,
            9,
        )
        self.assertEqual(strategy.low_count, 4)
        self.assertEqual(strategy.max_shared_red_balls, 4)
        self.assertEqual(strategy.rejection_lib_size, 500_000)


if __name__ == '__main__':
    unittest.main()
