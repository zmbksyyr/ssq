"""Canonical red-ball rule definitions and their score-budget constants."""

from ssq_rule_functions import (
    filter_ac_value,
    filter_all_cold,
    filter_big_small_ratio,
    filter_consecutive_numbers,
    filter_diagonal_consecutive,
    filter_ending_digits,
    filter_head_tail_range,
    filter_highly_regular,
    filter_modulo3_roads,
    filter_odd_even_ratio,
    filter_prime_composite_ratio,
    filter_recent_overlap,
    filter_related_numbers,
    filter_span,
    filter_sum_of_tails,
    filter_sum_value,
    filter_zones,
    score_big_small_balance,
    score_odd_even_balance,
    score_prime_balance,
    score_zone_balance,
)
from ssq_rule_models import RuleDefinition
from ssq_rule_validation import validate_rule_registry

COMBINATION_SIGNAL_WEIGHT = 0.50

RED_RULES = (
    RuleDefinition('highly_regular', True, lambda c, _: filter_highly_regular(c)),
    RuleDefinition('sum_value', True, lambda c, _: filter_sum_value(c)),
    RuleDefinition('span', True, lambda c, _: filter_span(c)),
    RuleDefinition(
        'consecutive_numbers', True,
        lambda c, _: filter_consecutive_numbers(c),
    ),
    RuleDefinition(
        'zones', True, lambda c, _: filter_zones(c), 0.10,
        lambda c, _: score_zone_balance(c),
    ),
    RuleDefinition(
        'ac_value', False, lambda c, _: filter_ac_value(c), 0.05,
        lambda c, _: float(filter_ac_value(c)),
    ),
    RuleDefinition(
        'prime_composite_ratio', False,
        lambda c, _: filter_prime_composite_ratio(c), 0.08,
        lambda c, _: score_prime_balance(c),
    ),
    RuleDefinition(
        'big_small_ratio', False,
        lambda c, _: filter_big_small_ratio(c), 0.08,
        lambda c, _: score_big_small_balance(c),
    ),
    RuleDefinition(
        'recent_overlap', True,
        lambda c, context: filter_recent_overlap(c, context.recent_draws),
    ),
    RuleDefinition(
        'all_cold', True,
        lambda c, context: filter_all_cold(c, context.omission_values),
    ),
    RuleDefinition(
        'odd_even_ratio', False, lambda c, _: filter_odd_even_ratio(c), 0.10,
        lambda c, _: score_odd_even_balance(c),
    ),
    RuleDefinition(
        'modulo3_roads', False,
        lambda c, _: filter_modulo3_roads(c), 0.04,
        lambda c, _: float(filter_modulo3_roads(c)),
    ),
    RuleDefinition('ending_digits', True, lambda c, _: filter_ending_digits(c)),
    RuleDefinition(
        'head_tail_range', False,
        lambda c, _: filter_head_tail_range(c), 0.03,
        lambda c, _: float(filter_head_tail_range(c)),
    ),
    RuleDefinition('sum_of_tails', True, lambda c, _: filter_sum_of_tails(c)),
    RuleDefinition(
        'related_numbers', True,
        lambda c, context: filter_related_numbers(
            c, context.last_draw or frozenset()
        ),
    ),
    RuleDefinition(
        'diagonal_consecutive', False,
        lambda c, context: filter_diagonal_consecutive(
            c, context.last_draw or (), context.previous_draw or ()
        ),
        0.02,
        lambda c, context: (
            1.0 if context.last_draw is None or context.previous_draw is None
            else float(filter_diagonal_consecutive(
                c, context.last_draw, context.previous_draw
            ))
        ),
    ),
)

validate_rule_registry(RED_RULES, COMBINATION_SIGNAL_WEIGHT)
FILTER_NAMES = tuple(rule.name for rule in RED_RULES)
HARD_FILTER_NAMES = tuple(rule.name for rule in RED_RULES if rule.hard)
SOFT_FILTER_NAMES = tuple(rule.name for rule in RED_RULES if not rule.hard)
