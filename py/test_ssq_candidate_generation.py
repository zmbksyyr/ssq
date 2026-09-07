import ast
import unittest
from pathlib import Path
from unittest.mock import Mock

import ssq_candidate_generation as generation
from ssq_config import StrategyConfig
from ssq_rule_models import RuleContext
from ssq_selection_models import CandidateGenerationRequest


class CandidateGenerationTests(unittest.TestCase):
    def test_pipeline_uses_injected_boundaries(self):
        request = CandidateGenerationRequest(
            red_scores={ball: float(ball) for ball in range(1, 34)},
            context=RuleContext(),
            rejection_set={(1, 2, 3, 4, 5, 6)},
            config=StrategyConfig(
                pool_size_red=7,
                high_count=2,
                low_count=2,
                recommendation_count=2,
            ),
        )
        validate = Mock()
        build_pool = Mock(return_value=range(1, 8))
        passes_filters = Mock(
            side_effect=lambda combo, *_: combo not in request.rejection_set
        )
        select_portfolio = Mock(
            side_effect=lambda selection: tuple(selection.passed_combos)[:2]
        )
        dependencies = generation.CandidateGenerationDependencies(
            validate_request=validate,
            build_pool=build_pool,
            passes_filters=passes_filters,
            select_portfolio=select_portfolio,
        )

        result = generation.generate_candidates(request, dependencies)

        validate.assert_called_once_with(request)
        build_pool.assert_called_once_with(
            request.red_scores,
            config=request.config,
            mode='mixed',
        )
        self.assertEqual(len(result.potential_combos), 7)
        self.assertEqual(len(result.passed_combos), 6)
        self.assertEqual(result.recommendations, result.passed_combos[:2])
        self.assertEqual(passes_filters.call_count, 7)

    def test_production_workflows_do_not_import_candidates_facade(self):
        for filename in (
            'ssq_backtest_evaluation.py',
            'ssq_prediction_workflow.py',
            'ssq_workflow.py',
        ):
            tree = ast.parse(
                Path(__file__).with_name(filename).read_text(encoding='utf-8')
            )
            imports = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            with self.subTest(filename=filename):
                self.assertNotIn('ssq_candidates', imports)


if __name__ == '__main__':
    unittest.main()
