import ast
import io
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).parent))
import ssq_backtest_runner as runner
import ssq_backtesting as backtesting
from ssq_backtest_models import BacktestRequest
from ssq_config import StrategyConfig


def make_dependencies(num_periods=2, pool_modes=('mixed',)):
    progress = Mock()
    return runner.BacktestRunnerDependencies(
        validate_request=Mock(return_value=(num_periods, pool_modes)),
        validate_params=Mock(return_value={'validated': True}),
        build_rank_band_widths=Mock(return_value={'high': 4}),
        prepare_issue=Mock(),
        evaluate_mode=Mock(),
        progress_factory=Mock(return_value=nullcontext(progress)),
    )


class BacktestRunnerTests(unittest.TestCase):
    def test_runtime_modules_bypass_backtesting_compatibility_facade(self):
        module_dir = Path(__file__).parent
        compatibility_modules = {'ssq_analyzer.py', 'ssq_backtesting.py'}
        offenders = []
        for path in module_dir.glob('ssq_*.py'):
            if path.name in compatibility_modules:
                continue
            tree = ast.parse(path.read_text(encoding='utf-8'))
            if any(
                isinstance(node, (ast.Import, ast.ImportFrom))
                and (
                    getattr(node, 'module', None) == 'ssq_backtesting'
                    or any(alias.name == 'ssq_backtesting' for alias in node.names)
                )
                for node in ast.walk(tree)
            ):
                offenders.append(path.name)

        self.assertEqual(offenders, [])

    def test_compatibility_wrapper_binds_legacy_patch_points(self):
        with patch.object(
            runner,
            'run_backtest',
            return_value='result',
        ) as run:
            result = backtesting.run_backtest('frame', 'request')

        self.assertEqual(result, 'result')
        dependencies = run.call_args.args[2]
        self.assertIs(dependencies.prepare_issue, backtesting.prepare_backtest_issue)
        self.assertIs(dependencies.evaluate_mode, backtesting.evaluate_backtest_mode)
        self.assertIs(dependencies.validate_request, backtesting.validate_backtest_request)

    def test_short_history_returns_empty_results_without_preparation(self):
        dependencies = make_dependencies(pool_modes=('mixed', 'low'))
        request = BacktestRequest({}, (), 2, ('mixed', 'low'), StrategyConfig())

        with patch('sys.stdout', new_callable=io.StringIO):
            result = runner.run_backtest([None] * 51, request, dependencies)

        self.assertEqual(set(result), {'mixed', 'low'})
        self.assertTrue(all(value.periods == 0 for value in result.values()))
        dependencies.prepare_issue.assert_not_called()
        dependencies.progress_factory.assert_not_called()

    def test_runner_processes_earlier_and_recent_windows_in_order(self):
        dependencies = make_dependencies()
        dependencies.prepare_issue.side_effect = (
            SimpleNamespace(issue='issue-1', selection_inputs='inputs-1'),
            SimpleNamespace(issue='issue-2', selection_inputs='inputs-2'),
        )
        request = BacktestRequest({}, ('feature',), 2, config=StrategyConfig())
        frame = [None] * 52

        with patch('sys.stdout', new_callable=io.StringIO):
            result = runner.run_backtest(frame, request, dependencies)

        self.assertEqual(
            [item.args[1] for item in dependencies.prepare_issue.call_args_list],
            [50, 51],
        )
        self.assertEqual(
            dependencies.evaluate_mode.call_args_list[0].args[2:],
            ('issue-1', 'inputs-1'),
        )
        self.assertEqual(
            dependencies.evaluate_mode.call_args_list[1].args[2:],
            ('issue-2', 'inputs-2'),
        )
        progress = dependencies.progress_factory.return_value.enter_result
        self.assertEqual(progress.mock_calls, [call.update(1), call.update(1)])
        self.assertEqual(set(result['mixed'].windows), {'earlier', 'recent'})


if __name__ == '__main__':
    unittest.main()
