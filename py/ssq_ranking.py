"""Combination scoring and diverse recommendation portfolio selection."""

from collections.abc import Sequence

import ssq_recommendation_validation as _recommendation_validation
from ssq_rule_models import (
    CombinationScoreContext,
    RuleContext,
    RuleDefinition,
)
from ssq_rule_registry import COMBINATION_SIGNAL_WEIGHT, RED_RULES

validate_recommendation_request = (
    _recommendation_validation.validate_recommendation_request
)


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


def score_combination(
    combo,
    scoring_context,
    rule_definitions: Sequence[RuleDefinition] = RED_RULES,
):
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
        for rule in rule_definitions if rule.scorer is not None
    )
    return COMBINATION_SIGNAL_WEIGHT * signal + rule_score


def score_red_combination(
    combo,
    red_scores,
    last_draw=None,
    previous_draw=None,
    rank_center_scores=None,
    context=None,
    rule_definitions: Sequence[RuleDefinition] = RED_RULES,
):
    """Score a combination with individual compatibility-style arguments."""
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
        rule_definitions,
    )


def select_recommendation_portfolio(
    request,
    rule_definitions: Sequence[RuleDefinition] = RED_RULES,
):
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
            -score_combination(combo, scoring_context, rule_definitions),
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
            if all(
                len(candidate & previous) <= overlap_limit
                for previous in selected_sets
            ):
                selected.append(combo)
                selected_sets.append(candidate)
                if len(selected) == limit:
                    return selected
    return selected
