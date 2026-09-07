import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ssq_backtest_validation as validation
import ssq_backtesting as backtesting
from ssq_config import StrategyConfig


class BacktestValidationTests(unittest.TestCase):
    def test_backtesting_preserves_validation_compatibility_export(self):
        self.assertIs(
            backtesting.validate_backtest_request,
            validation.validate_backtest_request,
        )

    def test_valid_controls_are_normalized(self):
        config = StrategyConfig()

        self.assertEqual(
            validation.validate_backtest_request(2, ['mixed', 'middle'], config),
            (2, ('mixed', 'middle')),
        )

    def test_invalid_controls_are_rejected_before_execution(self):
        config = StrategyConfig()
        cases = (
            (-1, ('mixed',), config),
            (1, (), config),
            (1, ('mixed', 'mixed'), config),
            (1, ('unknown',), config),
            (1, 'mixed', config),
            (1, ('mixed',), object()),
        )

        for periods, modes, candidate_config in cases:
            with self.subTest(periods=periods, modes=modes), self.assertRaises(
                (TypeError, ValueError)
            ):
                validation.validate_backtest_request(
                    periods,
                    modes,
                    candidate_config,
                )


if __name__ == '__main__':
    unittest.main()
