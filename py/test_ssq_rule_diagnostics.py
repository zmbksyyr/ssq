import unittest

import ssq_rule_diagnostics as diagnostics
import ssq_rule_execution as execution
from ssq_rule_models import RuleContext, RuleDefinition


class RuleDiagnosticTests(unittest.TestCase):
    def test_execution_preserves_diagnostic_compatibility_exports(self):
        self.assertIs(
            execution.explain_filter_failures,
            diagnostics.explain_filter_failures,
        )
        self.assertIs(
            execution.filter_pipeline_stats,
            diagnostics.filter_pipeline_stats,
        )

    def test_diagnostics_evaluate_soft_rules_without_filtering_them(self):
        rules = (
            RuleDefinition('hard', True, lambda combo, _: combo[0] > 0),
            RuleDefinition(
                'soft',
                False,
                lambda combo, _: combo[0] < 3,
                0.1,
                lambda _combo, _context: 1.0,
            ),
        )
        context = RuleContext()

        failures = diagnostics.explain_filter_failures(
            (4,),
            context,
            rules,
        )
        stats = diagnostics.filter_pipeline_stats(
            ((-1,), (4,)),
            context,
            rules,
        )

        self.assertEqual(failures, ['soft'])
        self.assertEqual(
            [item['rule'] for item in stats],
            ['hard', 'anti_crowding'],
        )


if __name__ == '__main__':
    unittest.main()
