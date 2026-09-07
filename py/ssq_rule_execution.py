"""Execution and diagnostics for a configured red-ball rule registry."""

from ssq_rule_diagnostics import explain_filter_failures, filter_pipeline_stats

__all__ = [
    'explain_filter_failures',
    'filter_pipeline_stats',
    'passes_red_filters',
]


def passes_red_filters(combo, context, rule_definitions, rejection_set=None):
    passes_rules = all(
        not rule.hard or rule.evaluator(combo, context)
        for rule in rule_definitions
    )
    return passes_rules and (rejection_set is None or combo not in rejection_set)
