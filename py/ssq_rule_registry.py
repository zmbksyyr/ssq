"""Red-ball rule definitions, registry validation, and filtering."""

from math import isclose, isfinite
from numbers import Real

import ssq_rule_functions as _rule_functions
from ssq_rule_models import RuleDefinition

COMBINATION_SIGNAL_WEIGHT = 0.50


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
score_zone_balance = _rule_functions.score_zone_balance
score_odd_even_balance = _rule_functions.score_odd_even_balance
score_prime_balance = _rule_functions.score_prime_balance
score_big_small_balance = _rule_functions.score_big_small_balance


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


def validate_signal_weight(signal_weight):
    if (
        isinstance(signal_weight, bool)
        or not isinstance(signal_weight, Real)
        or not isfinite(signal_weight)
        or not 0 <= signal_weight <= 1
    ):
        raise ValueError('基础排名信号权重必须为 0 到 1 之间的有限数值')


def validate_rule_names(rule_definitions):
    names = [rule.name for rule in rule_definitions]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError('规则名称必须为非空字符串')
    if len(names) != len(set(names)):
        raise ValueError('规则名称不能重复')


def validate_rule_definition(rule):
    if not isinstance(rule.hard, bool):
        raise TypeError(f'规则 {rule.name} 的 hard 标记必须为布尔值')
    if not callable(rule.evaluator):
        raise TypeError(f'规则 {rule.name} 的 evaluator 必须可调用')
    weight = rule.score_weight
    if (
        isinstance(weight, bool)
        or not isinstance(weight, Real)
        or not isfinite(weight)
        or weight < 0
    ):
        raise ValueError(f'规则 {rule.name} 的评分权重必须为有限非负数')
    if rule.scorer is None and weight != 0:
        raise ValueError(f'规则 {rule.name} 有评分权重但缺少 scorer')
    if rule.scorer is not None and (not callable(rule.scorer) or weight == 0):
        raise ValueError(f'规则 {rule.name} 的 scorer 与评分权重不一致')
    if not rule.hard and rule.scorer is None:
        raise ValueError(f'软规则 {rule.name} 必须提供 scorer')


def validate_rule_registry(rule_definitions, signal_weight=COMBINATION_SIGNAL_WEIGHT):
    """Validate rule identity, behavior, and the normalized score budget."""
    rule_definitions = tuple(rule_definitions)
    validate_signal_weight(signal_weight)
    validate_rule_names(rule_definitions)
    for rule in rule_definitions:
        validate_rule_definition(rule)

    score_weight_total = sum(
        (rule.score_weight for rule in rule_definitions),
        start=signal_weight,
    )
    if not isclose(score_weight_total, 1.0):
        raise ValueError(f'组合评分总权重必须为 1，实际为 {score_weight_total}')
    return tuple(rule_definitions)


validate_rule_registry(RED_RULES)
FILTER_NAMES = tuple(rule.name for rule in RED_RULES)
HARD_FILTER_NAMES = tuple(rule.name for rule in RED_RULES if rule.hard)
SOFT_FILTER_NAMES = tuple(rule.name for rule in RED_RULES if not rule.hard)


def passes_red_filters(combo, context, rejection_set=None):
    passes_rules = all(
        not rule.hard or rule.evaluator(combo, context)
        for rule in RED_RULES
    )
    return passes_rules and (rejection_set is None or combo not in rejection_set)


def explain_filter_failures(combo, context, rejection_set=None):
    failures = [
        rule.name for rule in RED_RULES
        if not rule.evaluator(combo, context)
    ]
    if rejection_set is not None and combo in rejection_set:
        failures.append('anti_crowding')
    return failures


def filter_pipeline_stats(combos, context, rejection_set=None):
    checks = [
        (rule.name, lambda combo, current=rule: current.evaluator(combo, context))
        for rule in RED_RULES if rule.hard
    ]
    checks.append((
        'anti_crowding',
        lambda combo: rejection_set is None or combo not in rejection_set,
    ))
    remaining = list(combos)
    stats = []
    for name, check in checks:
        before = len(remaining)
        remaining = [combo for combo in remaining if check(combo)]
        stats.append({
            'rule': name,
            'before': before,
            'removed': before - len(remaining),
            'remaining': len(remaining),
        })
    return stats
