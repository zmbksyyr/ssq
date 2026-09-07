"""Validation for red-ball candidate generation requests."""

from collections.abc import Collection, Mapping

from ssq_config import RED_POOL_MODES, StrategyConfig
from ssq_rule_models import RuleContext
from ssq_selection_models import CandidateGenerationRequest


def validate_candidate_generation_request(request):
    """Validate candidate-generation controls before enumerating combinations."""
    if not isinstance(request, CandidateGenerationRequest):
        raise TypeError('request must be a CandidateGenerationRequest')
    if not isinstance(request.context, RuleContext):
        raise TypeError('context must be a RuleContext')
    if not isinstance(request.config, StrategyConfig):
        raise TypeError('config must be a StrategyConfig')
    if not isinstance(request.red_scores, Mapping):
        raise TypeError('red_scores must be a mapping')
    if request.mode not in RED_POOL_MODES:
        raise ValueError(f'unknown pool mode: {request.mode}')
    if not isinstance(request.show_progress, bool):
        raise TypeError('show_progress must be a bool')
    if request.rejection_set is not None and (
        not isinstance(request.rejection_set, Collection)
        or isinstance(request.rejection_set, (str, bytes))
    ):
        raise TypeError('rejection_set must be a collection or None')
