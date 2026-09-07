import io
import unittest
from unittest.mock import patch

import ssq_cli
import ssq_config as config
from ssq_config_models import AnalyzerOptions


class CliTests(unittest.TestCase):
    def test_config_preserves_cli_compatibility_exports(self):
        self.assertIs(config.build_argument_parser, ssq_cli.build_argument_parser)
        self.assertIs(config.parse_cli_options, ssq_cli.parse_cli_options)

    def test_parser_builds_validated_runtime_options(self):
        options = ssq_cli.parse_cli_options([
            '--backtest-periods', '12',
            '--rejection-size', '500000',
            '--seed', '7',
            '--pool-mode', 'middle',
            '--compare-pools',
            '--non-interactive',
            '--rule-audit-periods', '50',
        ])

        self.assertIsInstance(options, AnalyzerOptions)
        self.assertEqual(options.backtest_periods, 12)
        self.assertEqual(options.rejection_size, 500_000)
        self.assertEqual(options.seed, 7)
        self.assertEqual(options.pool_mode, 'middle')
        self.assertTrue(options.compare_pools)
        self.assertTrue(options.non_interactive)
        self.assertEqual(options.rule_audit_periods, 50)

    def test_parser_reports_model_validation_errors(self):
        with (
            patch('sys.stderr', new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            ssq_cli.parse_cli_options(['--backtest-periods', '-1'])


if __name__ == '__main__':
    unittest.main()
