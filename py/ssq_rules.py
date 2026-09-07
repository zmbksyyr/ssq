"""Compatibility facade for rule filtering and combination ranking."""

import ssq_ranking as _ranking
from ssq_config import MAX_SHARED_RED_BALLS, NUM_RECOMMENDATIONS
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

build_combination_score_context = _ranking.build_combination_score_context
build_rank_center_scores = _ranking.build_rank_center_scores
score_rank_center_preference = _ranking.score_rank_center_preference
validate_recommendation_request = _ranking.validate_recommendation_request


def score_combination(combo, scoring_context):
    """Compatibility wrapper using this module's active rule registry."""
    return _ranking.score_combination(combo, scoring_context, RED_RULES)


def score_red_combination(
    combo,
    red_scores,
    last_draw=None,
    previous_draw=None,
    rank_center_scores=None,
    context=None,
):
    """Compatibility wrapper for scoring with individual arguments."""
    return _ranking.score_red_combination(
        combo,
        red_scores,
        last_draw=last_draw,
        previous_draw=previous_draw,
        rank_center_scores=rank_center_scores,
        context=context,
        rule_definitions=RED_RULES,
    )


def select_recommendation_portfolio(request):
    """Compatibility wrapper using this module's active rule registry."""
    return _ranking.select_recommendation_portfolio(request, RED_RULES)


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
