"""Diverse portfolio selection from scored red-ball combinations."""

from collections.abc import Sequence

from ssq_combination_scoring import (
    build_combination_score_context,
    score_combination,
)
from ssq_recommendation_validation import validate_recommendation_request
from ssq_rule_models import RuleDefinition
from ssq_rule_registry import RED_RULES


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
