"""Compatibility facade for red-ball rule definitions and execution."""

import ssq_rule_execution as _rule_execution
import ssq_rule_functions as _rule_functions
import ssq_rule_validation as _rule_validation
from ssq_rule_definitions import (
    COMBINATION_SIGNAL_WEIGHT,
    FILTER_NAMES,
    HARD_FILTER_NAMES,
    RED_RULES,
    SOFT_FILTER_NAMES,
)
from ssq_rule_scoring import (
    score_big_small_balance,
    score_odd_even_balance,
    score_prime_balance,
    score_zone_balance,
)

__all__ = [
    'COMBINATION_SIGNAL_WEIGHT',
    'FILTER_NAMES',
    'HARD_FILTER_NAMES',
    'RED_RULES',
    'SOFT_FILTER_NAMES',
    'score_big_small_balance',
    'score_odd_even_balance',
    'score_prime_balance',
    'score_zone_balance',
]

is_prime = _rule_functions.is_prime
calculate_ac_value = _rule_functions.calculate_ac_value
filter_highly_regular = _rule_functions.filter_highly_regular
filter_sum_value = _rule_functions.filter_sum_value
filter_span = _rule_functions.filter_span
filter_consecutive_numbers = _rule_functions.filter_consecutive_numbers
filter_zones = _rule_functions.filter_zones
filter_ac_value = _rule_functions.filter_ac_value
filter_prime_composite_ratio = _rule_functions.filter_prime_composite_ratio
filter_big_small_ratio = _rule_functions.filter_big_small_ratio
filter_recent_overlap = _rule_functions.filter_recent_overlap
filter_all_cold = _rule_functions.filter_all_cold
filter_odd_even_ratio = _rule_functions.filter_odd_even_ratio
filter_modulo3_roads = _rule_functions.filter_modulo3_roads
filter_ending_digits = _rule_functions.filter_ending_digits
filter_head_tail_range = _rule_functions.filter_head_tail_range
filter_sum_of_tails = _rule_functions.filter_sum_of_tails
filter_related_numbers = _rule_functions.filter_related_numbers
filter_diagonal_consecutive = _rule_functions.filter_diagonal_consecutive

validate_signal_weight = _rule_validation.validate_signal_weight
validate_rule_names = _rule_validation.validate_rule_names
validate_rule_definition = _rule_validation.validate_rule_definition


def validate_rule_registry(
    rule_definitions,
    signal_weight=COMBINATION_SIGNAL_WEIGHT,
):
    return _rule_validation.validate_rule_registry(rule_definitions, signal_weight)


def passes_red_filters(combo, context, rejection_set=None):
    return _rule_execution.passes_red_filters(
        combo,
        context,
        RED_RULES,
        rejection_set,
    )


def explain_filter_failures(combo, context, rejection_set=None):
    return _rule_execution.explain_filter_failures(
        combo,
        context,
        RED_RULES,
        rejection_set,
    )


def filter_pipeline_stats(combos, context, rejection_set=None):
    return _rule_execution.filter_pipeline_stats(
        combos,
        context,
        RED_RULES,
        rejection_set,
    )
