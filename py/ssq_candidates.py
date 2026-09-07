"""Red-ball pool construction, hard-rule filtering, and recommendations."""

from collections import Counter
from collections.abc import Collection, Mapping
from itertools import combinations

from ssq_config import DEFAULT_STRATEGY_CONFIG, RED_POOL_MODES, StrategyConfig
from ssq_core import validate_ball_scores
from ssq_domain import RED_BALLS
from ssq_rank_bands import RANK_BAND_NAMES, build_rank_bands
from ssq_ranking import select_recommendation_portfolio
from ssq_rule_models import RecommendationRequest, RuleContext
from ssq_rule_registry import passes_red_filters
from ssq_selection_models import CandidateGenerationRequest, RedCandidateSelection
from tqdm import tqdm


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


def count_actual_reds_by_rank_band(
    red_scores,
    actual_reds,
    config=DEFAULT_STRATEGY_CONFIG,
):
    """Count actual red balls by their model-score rank band."""
    red_scores = validate_ball_scores(red_scores, RED_BALLS, '红球')
    rank_bands = build_rank_bands(config)
    ranked = [
        ball for ball, _ in sorted(
            red_scores.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    rank_by_ball = {ball: rank for rank, ball in enumerate(ranked, 1)}
    counts = Counter({name: 0 for name in RANK_BAND_NAMES})
    for ball in actual_reds:
        rank = rank_by_ball[ball]
        band = next(
            (name for name, ranks in rank_bands.items() if rank in ranks),
            'other',
        )
        counts[band] += 1
    return counts


def build_red_pool(red_scores, config=DEFAULT_STRATEGY_CONFIG, mode='mixed'):
    """Build a red pool from one score band or a high/middle/low mixture."""
    red_scores = validate_ball_scores(red_scores, RED_BALLS, '红球')
    ranked = [
        ball for ball, _ in sorted(
            red_scores.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    if mode == 'high':
        return sorted(ranked[:config.pool_size_red])
    if mode == 'low':
        return sorted(ranked[-config.pool_size_red:])
    if mode == 'middle':
        start = max(0, (len(ranked) - config.pool_size_red) // 2)
        return sorted(ranked[start:start + config.pool_size_red])
    if mode != 'mixed':
        raise ValueError(f'unknown pool mode: {mode}')

    rank_bands = build_rank_bands(config)
    selected_ranks = (
        *rank_bands['high'],
        *rank_bands['middle'],
        *rank_bands['low'],
    )
    return sorted(ranked[rank - 1] for rank in selected_ranks)


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
