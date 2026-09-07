import io
import sys
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent))
import ssq_workflow as workflow


class AnalysisWorkflowTests(unittest.TestCase):
    def test_default_dependencies_bind_patchable_workflow_boundaries(self):
        dependencies = workflow.default_analysis_dependencies()

        self.assertIs(dependencies.prepare_history, workflow.prepare_history)
        self.assertIs(dependencies.evaluate_history, workflow.evaluate_history)
        self.assertIs(dependencies.train_models, workflow.train_final_models)
        self.assertIs(dependencies.select_current, workflow.select_current_issue)
        self.assertIs(
            dependencies.display_candidates,
            workflow.display_passed_combinations,
        )
        self.assertIs(dependencies.rank_duplex, workflow.rank_duplex_candidates)
        self.assertIs(
            dependencies.build_report_data,
            workflow.build_analysis_report_data,
        )
        self.assertIs(dependencies.now, workflow.local_now)
        self.assertIs(
            dependencies.collect_versions,
            workflow.collect_runtime_versions,
        )
        self.assertIs(dependencies.save_report, workflow.save_analysis_report)

    def test_run_composes_all_stages_through_explicit_dependencies(self):
        options = SimpleNamespace(non_interactive=True)
        history = object()
        evaluation = SimpleNamespace(
            loaded_params=SimpleNamespace(values={'weight': 1}),
        )
        selection = SimpleNamespace(
            passed_combos=((1, 2, 3, 4, 5, 6),),
            red_pool=(1, 2, 3, 4, 5, 6, 7),
        )
        current = SimpleNamespace(
            candidate_selection=selection,
            red_scores={1: 1.0},
            rule_context='rule-context',
        )
        generated_at = datetime(2026, 9, 8, tzinfo=timezone.utc)
        prepare_history = Mock(return_value=history)
        evaluate_history = Mock(return_value=evaluation)
        train_models = Mock(return_value=('red-models', 'blue-models'))
        select_current = Mock(return_value=current)
        display_candidates = Mock()
        rank_duplex = Mock(return_value=['duplex'])
        build_report_data = Mock(return_value='report-data')
        now = Mock(return_value=generated_at)
        collect_versions = Mock(return_value={'python': 'test'})
        save_report = Mock(return_value='report.txt')
        dependencies = replace(
            workflow.default_analysis_dependencies(),
            prepare_history=prepare_history,
            evaluate_history=evaluate_history,
            train_models=train_models,
            select_current=select_current,
            display_candidates=display_candidates,
            rank_duplex=rank_duplex,
            build_report_data=build_report_data,
            now=now,
            collect_versions=collect_versions,
            save_report=save_report,
        )

        with patch('sys.stdout', new_callable=io.StringIO):
            result = workflow.run_analysis(options, dependencies)

        self.assertEqual(result, 'report.txt')
        prepare_history.assert_called_once_with()
        evaluate_history.assert_called_once_with(history, options)
        train_models.assert_called_once_with(history)
        select_current.assert_called_once_with(
            history,
            options,
            evaluation.loaded_params.values,
            ('red-models', 'blue-models'),
        )
        display_candidates.assert_called_once_with(selection.passed_combos, True)
        duplex_request = rank_duplex.call_args.args[0]
        self.assertIs(duplex_request.passed_combos, selection.passed_combos)
        self.assertIs(duplex_request.red_pool, selection.red_pool)
        self.assertIs(duplex_request.red_scores, current.red_scores)
        self.assertEqual(duplex_request.context, 'rule-context')
        build_report_data.assert_called_once_with(
            history,
            evaluation,
            current,
            options,
            ['duplex'],
            generated_at,
            {'python': 'test'},
            workflow.MODEL_TRAINING_PARAMS,
        )
        save_report.assert_called_once_with('report-data')


if __name__ == '__main__':
    unittest.main()
