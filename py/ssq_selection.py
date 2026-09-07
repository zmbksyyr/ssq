"""Red-ball pool construction, candidate filtering, and ticket selection."""

import random
from collections import Counter
from collections.abc import Collection, Mapping
from itertools import combinations
from math import comb

from ssq_config import (
    DEFAULT_STRATEGY_CONFIG,
    RANDOM_SEED,
    RED_POOL_MODES,
    REJECTION_SEED_MULTIPLIER,
    TOTAL_RED_COMBINATIONS,
    StrategyConfig,
)
from ssq_core import (
    RED_BALLS,
    parse_issue,
    parse_red_balls,
    validate_ball_scores,
)
from ssq_rank_bands import (
    RANK_BAND_NAMES,
)
from ssq_rank_bands import (
    build_rank_band_labels as _build_rank_band_labels,
)
from ssq_rank_bands import (
    build_rank_band_widths as _build_rank_band_widths,
)
from ssq_rank_bands import (
    build_rank_bands as _build_rank_bands,
)
from ssq_ranking import (
    build_combination_score_context,
    score_combination,
    select_recommendation_portfolio,
)
from ssq_rule_models import RecommendationRequest, RuleContext
from ssq_rule_registry import passes_red_filters
from ssq_selection_models import (
    CandidateGenerationRequest,
    DuplexSelectionRequest,
    RedCandidateSelection,
)
from tqdm import tqdm


def build_rank_bands(config=DEFAULT_STRATEGY_CONFIG):
    """Compatibility wrapper for shared score-rank band definitions."""
    return _build_rank_bands(config)


def build_rank_band_widths(config=DEFAULT_STRATEGY_CONFIG):
    """Compatibility wrapper for shared score-rank band widths."""
    return _build_rank_band_widths(config)


def build_rank_band_labels(config=DEFAULT_STRATEGY_CONFIG):
    """Compatibility wrapper for shared score-rank band labels."""
    return _build_rank_band_labels(config)


RANK_BANDS = build_rank_bands()
RANK_BAND_WIDTHS = build_rank_band_widths()


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


def validate_duplex_selection_request(request):
    """Validate and materialize duplex inputs before exhaustive ranking."""
    if not isinstance(request, DuplexSelectionRequest):
        raise TypeError('request must be a DuplexSelectionRequest')
    if not isinstance(request.context, RuleContext):
        raise TypeError('context must be a RuleContext')
    if not isinstance(request.red_pool, Collection) or isinstance(
        request.red_pool, (str, bytes)
    ):
        raise TypeError('red_pool must be a collection')
    red_pool = tuple(parse_red_balls(
        request.red_pool,
        expected_count=len(request.red_pool),
    ))
    if len(red_pool) < 7:
        raise ValueError('red_pool must contain at least 7 balls')
    if not isinstance(request.passed_combos, Collection) or isinstance(
        request.passed_combos, (str, bytes)
    ):
        raise TypeError('passed_combos must be a collection')
    passed_combos = tuple(
        tuple(parse_red_balls(combo)) for combo in request.passed_combos
    )
    if len(passed_combos) != len(set(passed_combos)):
        raise ValueError('passed_combos cannot contain duplicates')
    if any(not set(combo).issubset(red_pool) for combo in passed_combos):
        raise ValueError('every passed combination must belong to red_pool')
    red_scores = (
        validate_ball_scores(request.red_scores, RED_BALLS, 'red ball')
        if request.red_scores is not None else None
    )
    return passed_combos, red_pool, red_scores


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


def make_rejection_set(size, rng=None):
    """Create a reproducible anti-crowding sample of six-red combinations."""
    if not 0 <= size <= TOTAL_RED_COMBINATIONS:
        raise ValueError(
            f'rejection size must be between 0 and {TOTAL_RED_COMBINATIONS}'
        )
    rng = rng or random.Random(RANDOM_SEED)
    rejection_set = set()
    while len(rejection_set) < size:
        rejection_set.add(tuple(sorted(rng.sample(RED_BALLS, 6))))
    return rejection_set


def rejection_seed_for_issue(base_seed, issue):
    """Derive a reproducible anti-crowding seed that changes every issue."""
    return int(base_seed) * REJECTION_SEED_MULTIPLIER + parse_issue(issue)


def rank_duplex_candidates(request):
    """Rank every 7-red ticket by valid subticket coverage, then strategy score."""
    passed_combos, red_pool, red_scores = validate_duplex_selection_request(request)
    if not passed_combos:
        return []

    passed_combos_set = set(passed_combos)
    scoring_context = (
        build_combination_score_context(
            red_scores,
            request.context,
        )
        if red_scores is not None else None
    )
    ranked = []
    seven_ball_combos = combinations(red_pool, 7)
    for seven_combo in tqdm(
        seven_ball_combos,
        total=comb(len(red_pool), 7),
        desc='生成7红球大底',
        leave=False,
        ncols=80,
    ):
        subtickets = list(combinations(seven_combo, 6))
        valid_subtickets = [
            subticket for subticket in subtickets
            if subticket in passed_combos_set
        ]
        coverage = len(valid_subtickets)
        if not coverage:
            continue
        quality = 0.0
        if scoring_context:
            quality = sum(
                score_combination(subticket, scoring_context)
                for subticket in valid_subtickets
            ) / coverage
        ranked.append((seven_combo, coverage, quality))
    ranked.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [(combo, coverage) for combo, coverage, _ in ranked]


def find_best_7_red_combinations(
    passed_combos_tuples,
    red_pool,
    red_scores=None,
    last_draw_set=None,
    last_2_draw_set=None,
    context=None,
):
    """Compatibility wrapper for request-based duplex selection."""
    return rank_duplex_candidates(DuplexSelectionRequest(
        passed_combos=passed_combos_tuples,
        red_pool=red_pool,
        red_scores=red_scores,
        context=context or RuleContext(
            last_draw=last_draw_set,
            previous_draw=last_2_draw_set,
        ),
    ))
