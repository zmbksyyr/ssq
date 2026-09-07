"""Validation and normalization for recommendation portfolio requests."""

from collections.abc import Mapping
from math import isfinite
from numbers import Real

from ssq_config import normalize_integer_param
from ssq_core import RED_BALLS, parse_red_balls
from ssq_rule_models import RecommendationRequest, RuleContext


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
