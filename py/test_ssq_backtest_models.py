import unittest
from collections import Counter

import ssq_backtest_models as models
import ssq_backtesting as backtesting
from ssq_config import DEFAULT_STRATEGY_CONFIG
from ssq_rule_models import RuleContext


class BacktestModelTests(unittest.TestCase):
    def test_backtesting_preserves_model_compatibility_exports(self):
        for name in (
            'BacktestIssue',
            'BacktestSelectionInputs',
            'BacktestRunContext',
            'BacktestRequest',
            'PreparedBacktestIssue',
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(backtesting, name), getattr(models, name))

    def test_prepared_issue_groups_draw_and_selection_inputs(self):
        issue = models.BacktestIssue(
            actual_reds=frozenset((1, 2, 3, 4, 5, 6)),
            actual_blue=7,
            recommended_blue=8,
            rank_band_hits=Counter({'middle': 6}),
        )
        selection_inputs = models.BacktestSelectionInputs(
            red_scores={1: 0.5},
            context=RuleContext(last_draw={1}),
            rejection_set=set(),
            config=DEFAULT_STRATEGY_CONFIG,
        )

        prepared = models.PreparedBacktestIssue(issue, selection_inputs)

        self.assertIs(prepared.issue, issue)
        self.assertIs(prepared.selection_inputs, selection_inputs)


if __name__ == '__main__':
    unittest.main()
