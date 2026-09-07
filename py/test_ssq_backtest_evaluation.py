import unittest
from collections import Counter
from types import SimpleNamespace

import ssq_backtest_evaluation as evaluation
import ssq_backtesting as backtesting
from ssq_backtest_metrics import BacktestAccumulator
from ssq_backtest_models import BacktestIssue


class BacktestEvaluationTests(unittest.TestCase):
    def test_backtesting_preserves_evaluation_compatibility_exports(self):
        self.assertIs(
            backtesting.evaluate_backtest_mode,
            evaluation.evaluate_backtest_mode,
        )
        self.assertIs(
            backtesting.record_backtest_selection,
            evaluation.record_backtest_selection,
        )

    def test_empty_selection_records_issue_level_metrics_without_tickets(self):
        issue = BacktestIssue(
            actual_reds=frozenset((1, 2, 3, 4, 5, 6)),
            actual_blue=7,
            recommended_blue=7,
            rank_band_hits=Counter({'middle': 6}),
        )
        selection = SimpleNamespace(
            red_pool=(1, 2, 3, 7, 8, 9),
            passed_combos=(),
            recommendations=(),
        )
        accumulator = BacktestAccumulator()

        hits = evaluation.record_backtest_selection(
            accumulator,
            selection,
            issue,
        )

        self.assertEqual(hits, {})
        self.assertEqual(accumulator.evaluated_periods, 1)
        self.assertEqual(accumulator.pool_red_hits, 3)
        self.assertEqual(accumulator.blue_hit_periods, 1)
        self.assertEqual(accumulator.rank_band_hits, {'middle': 6})
        self.assertEqual(accumulator.tickets, 0)
        self.assertEqual(accumulator.cost, 0)


if __name__ == '__main__':
    unittest.main()
