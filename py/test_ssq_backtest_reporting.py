import unittest

import ssq_backtest_reporting as backtest_reporting
import ssq_reporting as reporting


class BacktestReportingTests(unittest.TestCase):
    def test_reporting_preserves_backtest_format_compatibility_exports(self):
        names = (
            'format_strategy_parameters',
            'format_backtest_metrics',
            'format_pool_comparison',
            'format_window_stability',
            'format_rank_band_distribution',
            'format_prize_counts',
            'format_backtest_report',
        )

        for name in names:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(reporting, name),
                    getattr(backtest_reporting, name),
                )

    def test_prize_display_order_remains_complete(self):
        self.assertEqual(
            backtest_reporting.PRIZE_DISPLAY_ORDER,
            ('一等奖', '二等奖', '三等奖', '四等奖', '五等奖', '六等奖'),
        )


if __name__ == '__main__':
    unittest.main()
