from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from math import isclose, isfinite
from numbers import Real

from ssq_config import (
    MAX_SHARED_RED_BALLS,
    NUM_RECOMMENDATIONS,
    normalize_integer_param,
)
from ssq_core import PRIME_RED_BALLS, RED_BALLS, parse_red_balls

COMBINATION_SIGNAL_WEIGHT = 0.50


@dataclass(frozen=True)
class RuleContext:
    omission_values: Mapping[int, int] = field(default_factory=dict)
    recent_draws: Sequence[Collection[int]] = field(default_factory=tuple)
    last_draw: Collection[int] | None = None
    previous_draw: Collection[int] | None = None


@dataclass(frozen=True)
class RuleDefinition:
    name: str
    hard: bool
    evaluator: Callable[[tuple[int, ...], RuleContext], bool]
    score_weight: float = 0.0
    scorer: Callable[[tuple[int, ...], RuleContext], float] | None = None


@dataclass(frozen=True)
class CombinationScoreContext:
    red_scores: Mapping[int, float]
    rank_center_scores: Mapping[int, float]
    rule_context: RuleContext


@dataclass(frozen=True)
class RecommendationRequest:
    passed_combos: Iterable[tuple[int, ...]]
    red_scores: Mapping[int, float]
    context: RuleContext
    limit: int = NUM_RECOMMENDATIONS
    max_shared: int = MAX_SHARED_RED_BALLS


def validate_recommendation_request(request):
    """Validate and materialize recommendation inputs at the API boundary."""
    if not isinstance(request, RecommendationRequest):
        raise TypeError('request must be a RecommendationRequest')
    if not isinstance(request.context, RuleContext):
        raise TypeError('context must be a RuleContext')
    limit = normalize_integer_param('limit', request.limit)
    max_shared = normalize_integer_param('max_shared', request.max_shared)
    if limit < 0:
        raise ValueError('limit cannot be negative')
    if not 0 <= max_shared <= 6:
        raise ValueError('max_shared must be between 0 and 6')
    if not isinstance(request.red_scores, Mapping):
        raise TypeError('red_scores must be a mapping')
    if isinstance(request.passed_combos, (str, bytes)):
        raise TypeError('passed_combos must be an iterable of combinations')
    try:
        passed_combos = tuple(
            tuple(parse_red_balls(combo)) for combo in request.passed_combos
        )
    except TypeError as exc:
        raise TypeError('passed_combos must be an iterable of combinations') from exc
    if len(passed_combos) != len(set(passed_combos)):
        raise ValueError('passed_combos cannot contain duplicates')
    invalid_balls = [ball for ball in request.red_scores if ball not in RED_BALLS]
    if invalid_balls:
        raise ValueError(f'red_scores contains invalid balls: {invalid_balls}')
    required_balls = {ball for combo in passed_combos for ball in combo}
    missing_balls = sorted(required_balls - set(request.red_scores))
    if missing_balls:
        raise ValueError(f'red_scores is missing candidate balls: {missing_balls}')
    red_scores = {}
    for ball, value in request.red_scores.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f'red ball {ball} score must be numeric')
        value = float(value)
        if not isfinite(value):
            raise ValueError(f'red ball {ball} score must be finite')
        red_scores[ball] = value
    return passed_combos, red_scores, limit, max_shared


def is_prime(number):
    return number in PRIME_RED_BALLS


def calculate_ac_value(combo):
    standard_ac = len({abs(left - right) for left, right in combinations(combo, 2)})
    return standard_ac - 5


def filter_highly_regular(combo):
    return len({combo[index + 1] - combo[index] for index in range(len(combo) - 1)}) > 1


def filter_sum_value(combo):
    return 70 <= sum(combo) <= 160


def filter_span(combo):
    return combo[-1] - combo[0] >= 15


def filter_consecutive_numbers(combo):
    groups = 0
    max_length = 0
    current_length = 1
    for index in range(len(combo) - 1):
        if combo[index + 1] - combo[index] == 1:
            current_length += 1
        else:
            if current_length >= 2:
                groups += 1
                max_length = max(max_length, current_length)
            current_length = 1
    if current_length >= 2:
        groups += 1
        max_length = max(max_length, current_length)
    return not (groups >= 3 or max_length >= 4)


def filter_zones(combo):
    all_in_small = all(ball <= 11 for ball in combo)
    all_in_medium = all(12 <= ball <= 22 for ball in combo)
    all_in_large = all(ball >= 23 for ball in combo)
    return not (all_in_small or all_in_medium or all_in_large)


def filter_ac_value(combo):
    return 6 <= calculate_ac_value(combo) <= 10


def filter_prime_composite_ratio(combo):
    prime_count = sum(is_prime(ball) for ball in combo)
    return prime_count not in (0, 1, 5, 6)


def filter_big_small_ratio(combo):
    small_count = sum(ball <= 16 for ball in combo)
    return small_count not in (0, 1, 5, 6)


def filter_recent_overlap(combo, recent_draws):
    candidate = set(combo)
    return all(len(candidate & draw) < 4 for draw in recent_draws)


def filter_all_cold(combo, omission_values):
    return not all(omission_values.get(ball, 0) > 15 for ball in combo)


def filter_odd_even_ratio(combo):
    even_count = sum(ball % 2 == 0 for ball in combo)
    return even_count not in (0, 1, 5, 6)


def filter_modulo3_roads(combo):
    return len({ball % 3 for ball in combo}) == 3


def filter_ending_digits(combo):
    counts = Counter(ball % 10 for ball in combo)
    return max(counts.values()) < 3 and len(counts) > 2


def filter_head_tail_range(combo):
    return combo[0] <= 10 and combo[-1] >= 25


def filter_sum_of_tails(combo):
    return 15 <= sum(ball % 10 for ball in combo) <= 45


def filter_related_numbers(combo, last_draw):
    candidate = set(combo)
    repeats = candidate & last_draw
    adjacent_numbers = (
        {number - 1 for number in last_draw}
        | {number + 1 for number in last_draw}
    )
    return bool(repeats or candidate & adjacent_numbers)


def filter_diagonal_consecutive(combo, last_draw, previous_draw):
    return not any(
        ball - 1 in last_draw and ball - 2 in previous_draw
        for ball in combo
    )


def score_zone_balance(combo):
    counts = (
        sum(ball <= 11 for ball in combo),
        sum(12 <= ball <= 22 for ball in combo),
        sum(ball >= 23 for ball in combo),
    )
    return 1.0 - (max(counts) - min(counts)) / 6


def score_odd_even_balance(combo):
    return 1.0 - abs(sum(ball % 2 for ball in combo) - 3) / 3


def score_prime_balance(combo):
    return 1.0 - abs(sum(is_prime(ball) for ball in combo) - 3) / 3


def score_big_small_balance(combo):
    return 1.0 - abs(sum(ball <= 16 for ball in combo) - 3) / 3


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
    RuleDefinition(
        'sum_of_tails', True, lambda c, _: filter_sum_of_tails(c)
    ),
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


def build_rank_center_scores(red_scores):
    ranked = sorted(red_scores, key=lambda ball: (-red_scores[ball], ball))
    if len(ranked) <= 1:
        return {ball: 1.0 for ball in ranked}
    center = (len(ranked) - 1) / 2
    return {
        ball: 1.0 - abs(index - center) / center
        for index, ball in enumerate(ranked)
    }


def score_rank_center_preference(combo, red_scores, rank_center_scores=None):
    rank_scores = rank_center_scores or build_rank_center_scores(red_scores)
    return sum(rank_scores.get(ball, 0.0) for ball in combo) / len(combo)


def build_combination_score_context(red_scores, context=None, rank_center_scores=None):
    """Build reusable inputs for ranking multiple red-ball combinations."""
    return CombinationScoreContext(
        red_scores=red_scores,
        rank_center_scores=(
            rank_center_scores or build_rank_center_scores(red_scores)
        ),
        rule_context=context or RuleContext(),
    )


def score_combination(combo, scoring_context):
    """Score one combination from a precomputed ranking context."""
    if not isinstance(scoring_context, CombinationScoreContext):
        raise TypeError('scoring_context 必须为 CombinationScoreContext')
    signal = score_rank_center_preference(
        combo,
        scoring_context.red_scores,
        scoring_context.rank_center_scores,
    )
    rule_score = sum(
        rule.score_weight * rule.scorer(combo, scoring_context.rule_context)
        for rule in RED_RULES if rule.scorer is not None
    )
    return COMBINATION_SIGNAL_WEIGHT * signal + rule_score


def score_red_combination(combo, red_scores, last_draw=None, previous_draw=None,
                          rank_center_scores=None, context=None):
    """Compatibility wrapper for scoring with individual arguments."""
    return score_combination(
        combo,
        build_combination_score_context(
            red_scores,
            context=context or RuleContext(
                last_draw=last_draw,
                previous_draw=previous_draw,
            ),
            rank_center_scores=rank_center_scores,
        ),
    )


def select_recommendation_portfolio(request):
    """Select a diverse portfolio from ranked valid combinations."""
    passed_combos, red_scores, limit, max_shared = (
        validate_recommendation_request(request)
    )
    if limit == 0:
        return []
    scoring_context = build_combination_score_context(
        red_scores,
        request.context,
    )
    ranked = sorted(
        passed_combos,
        key=lambda combo: (
            -score_combination(combo, scoring_context),
            combo,
        ),
    )
    selected = []
    selected_sets = []
    for overlap_limit in range(max_shared, 7):
        for combo in ranked:
            if combo in selected:
                continue
            candidate = set(combo)
            if all(len(candidate & previous) <= overlap_limit
                   for previous in selected_sets):
                selected.append(combo)
                selected_sets.append(candidate)
                if len(selected) == limit:
                    return selected
    return selected


def select_recommendations(
    passed_combos,
    red_scores,
    last_draw=None,
    previous_draw=None,
    limit=NUM_RECOMMENDATIONS,
    max_shared=MAX_SHARED_RED_BALLS,
    context=None,
):
    """Compatibility wrapper for request-based portfolio selection."""
    return select_recommendation_portfolio(RecommendationRequest(
        passed_combos=passed_combos,
        red_scores=red_scores,
        context=context or RuleContext(
            last_draw=last_draw,
            previous_draw=previous_draw,
        ),
        limit=limit,
        max_shared=max_shared,
    ))
