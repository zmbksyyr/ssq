"""Dependency-driven red-ball candidate generation pipeline."""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

from ssq_candidate_validation import validate_candidate_generation_request
from ssq_ranking import select_recommendation_portfolio
from ssq_red_pool import build_red_pool
from ssq_rule_models import RecommendationRequest
from ssq_rule_registry import passes_red_filters
from ssq_selection_models import RedCandidateSelection
from tqdm import tqdm


@dataclass(frozen=True)
class CandidateGenerationDependencies:
    """Replaceable boundaries used while enumerating and selecting candidates."""

    validate_request: Callable = validate_candidate_generation_request
    build_pool: Callable = build_red_pool
    passes_filters: Callable = passes_red_filters
    select_portfolio: Callable = select_recommendation_portfolio
    progress: Callable = tqdm


def generate_candidates(request, dependencies=None):
    """Run the shared red-ball selection pipeline for live runs and backtests."""
    dependencies = dependencies or CandidateGenerationDependencies()
    dependencies.validate_request(request)
    red_pool = tuple(dependencies.build_pool(
        request.red_scores,
        config=request.config,
        mode=request.mode,
    ))
    potential_combos = tuple(combinations(red_pool, 6))
    iterator = (
        dependencies.progress(potential_combos, desc='规则过滤进度', ncols=80)
        if request.show_progress else potential_combos
    )
    passed_combos = tuple(
        combo for combo in iterator
        if dependencies.passes_filters(
            combo,
            request.context,
            request.rejection_set,
        )
    )
    recommendations = tuple(dependencies.select_portfolio(RecommendationRequest(
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
