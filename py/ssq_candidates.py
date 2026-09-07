"""Compatibility facade for red-ball candidate generation."""

from ssq_candidate_generation import CandidateGenerationDependencies
from ssq_candidate_generation import generate_candidates as _generate_candidates
from ssq_candidate_validation import validate_candidate_generation_request
from ssq_config import DEFAULT_STRATEGY_CONFIG
from ssq_ranking import select_recommendation_portfolio
from ssq_red_pool import build_red_pool, count_actual_reds_by_rank_band
from ssq_rule_registry import passes_red_filters
from ssq_selection_models import CandidateGenerationRequest
from tqdm import tqdm

__all__ = [
    'build_red_pool',
    'count_actual_reds_by_rank_band',
    'generate_candidates',
    'generate_red_candidates',
    'passes_red_filters',
    'select_recommendation_portfolio',
    'validate_candidate_generation_request',
]


def generate_candidates(request):
    """Run candidate generation through patchable legacy boundaries."""
    dependencies = CandidateGenerationDependencies(
        validate_request=validate_candidate_generation_request,
        build_pool=build_red_pool,
        passes_filters=passes_red_filters,
        select_portfolio=select_recommendation_portfolio,
        progress=tqdm,
    )
    return _generate_candidates(request, dependencies)


def generate_red_candidates(
    red_scores,
    context,
    rejection_set,
    config=DEFAULT_STRATEGY_CONFIG,
    mode='mixed',
    show_progress=False,
):
    """Compatibility wrapper for request-based candidate generation."""
    return generate_candidates(CandidateGenerationRequest(
        red_scores=red_scores,
        context=context,
        rejection_set=rejection_set,
        config=config,
        mode=mode,
        show_progress=show_progress,
    ))
