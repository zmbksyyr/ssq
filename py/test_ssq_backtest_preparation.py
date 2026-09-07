import unittest
from collections import Counter
from unittest.mock import Mock

import pandas as pd
import ssq_backtest_preparation as preparation
from ssq_backtest_models import BacktestRunContext
from ssq_config import StrategyConfig


class BacktestPreparationTests(unittest.TestCase):
    def test_insufficient_training_history_skips_before_training(self):
        frame = pd.DataFrame({'期号': range(24)})
        train = Mock()
        dependencies = preparation.BacktestPreparationDependencies(
            train_models=train,
            validate_models=Mock(),
            score_balls=Mock(),
            count_rank_bands=Mock(),
            build_rule_context=Mock(),
            derive_rejection_seed=Mock(),
            build_rejection_set=Mock(),
        )
        context = BacktestRunContext({}, (), StrategyConfig())

        result = preparation.prepare_backtest_issue(
            frame,
            24,
            context,
            dependencies,
        )

        self.assertIsNone(result)
        train.assert_not_called()

    def test_preparation_excludes_the_evaluated_issue(self):
        frame = pd.DataFrame({
            '期号': list(range(2025001, 2025027)),
            '红球': [[1, 2, 3, 4, 5, 6] for _ in range(26)],
            '蓝球': [1 for _ in range(25)] + [7],
        })
        red_models = {ball: object() for ball in range(1, 34)}
        blue_models = {ball: object() for ball in range(1, 17)}
        red_scores = {ball: float(ball) for ball in range(1, 34)}
        blue_scores = {ball: float(ball == 7) for ball in range(1, 17)}
        train = Mock(return_value=(red_models, blue_models))
        dependencies = preparation.BacktestPreparationDependencies(
            train_models=train,
            validate_models=Mock(),
            score_balls=Mock(return_value=(red_scores, blue_scores)),
            count_rank_bands=Mock(return_value=Counter({'middle': 6})),
            build_rule_context=Mock(return_value='rule context'),
            derive_rejection_seed=Mock(return_value=123),
            build_rejection_set=Mock(return_value=set()),
        )
        config = StrategyConfig(rejection_lib_size=0, random_seed=9)
        context = BacktestRunContext({'weight': 1}, ('feature',), config)

        result = preparation.prepare_backtest_issue(
            frame,
            25,
            context,
            dependencies,
        )

        training_frame = train.call_args.args[0]
        self.assertEqual(training_frame.iloc[-1]['期号'], 2025025)
        self.assertNotIn(2025026, training_frame['期号'].tolist())
        self.assertEqual(result.issue.actual_blue, 7)
        self.assertEqual(result.issue.recommended_blue, 7)
        self.assertEqual(result.selection_inputs.red_scores, red_scores)
        self.assertEqual(result.selection_inputs.context, 'rule context')
        dependencies.derive_rejection_seed.assert_called_once_with(9, 2025026)


if __name__ == '__main__':
    unittest.main()
