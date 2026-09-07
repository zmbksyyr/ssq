"""Red-ball pool construction, hard-rule filtering, and recommendations."""

from itertools import combinations

from ssq_candidate_validation import validate_candidate_generation_request
from ssq_config import DEFAULT_STRATEGY_CONFIG
from ssq_ranking import select_recommendation_portfolio
from ssq_red_pool import build_red_pool, count_actual_reds_by_rank_band
from ssq_rule_models import RecommendationRequest
from ssq_rule_registry import passes_red_filters
from ssq_selection_models import CandidateGenerationRequest, RedCandidateSelection
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
    """Run the shared red-ball selection pipeline for live runs and backtests."""
    validate_candidate_generation_request(request)
    red_pool = tuple(build_red_pool(
        request.red_scores,
        config=request.config,
        mode=request.mode,
    ))
    potential_combos = tuple(combinations(red_pool, 6))
    iterator = (
        tqdm(potential_combos, desc='规则过滤进度', ncols=80)
        if request.show_progress else potential_combos
    )
    passed_combos = tuple(
        combo for combo in iterator
        if passes_red_filters(combo, request.context, request.rejection_set)
    )
    recommendations = tuple(select_recommendation_portfolio(RecommendationRequest(
        passed_combos=passed_combos,
        red_scores=request.red_scores,
        context=request.context,
        limit=request.config.recommendation_count,
        max_shared=request.config.max_shared_red_balls,
    )))
    return RedCandidateSelection(
        red_pool=red_pool,
        potential_combos=potential_combos,
        passed_combos=passed_combos,
        recommendations=recommendations,
    )


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
