import unittest

import ssq_rule_execution as execution
from ssq_rule_models import RuleContext, RuleDefinition


class RuleExecutionTests(unittest.TestCase):
    def setUp(self):
        self.context = RuleContext()
        self.rules = (
            RuleDefinition('positive', True, lambda combo, _context: combo[0] > 0),
            RuleDefinition('even', True, lambda combo, _context: combo[0] % 2 == 0),
            RuleDefinition(
                'soft',
                False,
                lambda combo, _context: combo[0] < 3,
                0.1,
                lambda _combo, _context: 1.0,
            ),
        )

    def test_hard_filtering_ignores_soft_rule_for_admission(self):
        self.assertTrue(
            execution.passes_red_filters((4,), self.context, self.rules)
        )
        self.assertFalse(
            execution.passes_red_filters((3,), self.context, self.rules)
        )

    def test_explanation_includes_soft_rules_and_anti_crowding(self):
        failures = execution.explain_filter_failures(
            (4,),
            self.context,
            self.rules,
            {(4,)},
        )

        self.assertEqual(failures, ['soft', 'anti_crowding'])

    def test_pipeline_statistics_are_incremental_and_hard_only(self):
        stats = execution.filter_pipeline_stats(
            ((-2,), (2,), (3,), (4,)),
            self.context,
            self.rules,
            {(4,)},
        )

        self.assertEqual([item['rule'] for item in stats], [
            'positive', 'even', 'anti_crowding',
        ])
        self.assertEqual(
            [(item['before'], item['removed'], item['remaining']) for item in stats],
            [(4, 1, 3), (3, 1, 2), (2, 1, 1)],
        )


if __name__ == '__main__':
    unittest.main()
