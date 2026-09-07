"""Deterministic scoring for red-ball combinations."""

from collections.abc import Sequence

from ssq_rule_models import (
    CombinationScoreContext,
    RuleContext,
    RuleDefinition,
)
from ssq_rule_registry import COMBINATION_SIGNAL_WEIGHT, RED_RULES


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
