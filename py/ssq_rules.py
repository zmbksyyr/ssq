"""Compatibility facade for red-ball filtering and combination ranking."""

from collections.abc import Mapping
from math import isfinite
from numbers import Real

from ssq_config import (
    MAX_SHARED_RED_BALLS,
    NUM_RECOMMENDATIONS,
    normalize_integer_param,
)
from ssq_core import RED_BALLS, parse_red_balls
from ssq_rule_models import (  # noqa: F401 - compatibility exports
    CombinationScoreContext,
    RecommendationRequest,
    RuleContext,
    RuleDefinition,
)
from ssq_rule_registry import (  # noqa: F401 - compatibility exports
    COMBINATION_SIGNAL_WEIGHT,
    FILTER_NAMES,
    HARD_FILTER_NAMES,
    RED_RULES,
    SOFT_FILTER_NAMES,
    calculate_ac_value,
    explain_filter_failures,
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
    filter_pipeline_stats,
    filter_prime_composite_ratio,
    filter_recent_overlap,
    filter_related_numbers,
    filter_span,
    filter_sum_of_tails,
    filter_sum_value,
    filter_zones,
    is_prime,
    passes_red_filters,
    score_big_small_balance,
    score_odd_even_balance,
    score_prime_balance,
    score_zone_balance,
    validate_rule_definition,
    validate_rule_names,
    validate_rule_registry,
    validate_signal_weight,
)


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


def build_combination_score_context(
    red_scores,
    context=None,
    rank_center_scores=None,
):
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
        raise TypeError('scoring_context must be a CombinationScoreContext')
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


def score_red_combination(
    combo,
    red_scores,
    last_draw=None,
    previous_draw=None,
    rank_center_scores=None,
    context=None,
):
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
        key=lambda combo: (-score_combination(combo, scoring_context), combo),
    )
    selected = []
    selected_sets = []
    for overlap_limit in range(max_shared, 7):
        for combo in ranked:
            if combo in selected:
                continue
            candidate = set(combo)
            if all(
                len(candidate & previous) <= overlap_limit
                for previous in selected_sets
            ):
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
