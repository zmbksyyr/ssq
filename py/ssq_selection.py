"""Red-ball pool construction, candidate filtering, and ticket selection."""

from collections.abc import Collection
from itertools import combinations
from math import comb

import ssq_anti_crowding as _anti_crowding
import ssq_candidates as _candidates
import ssq_selection_models as _selection_models
from ssq_config import DEFAULT_STRATEGY_CONFIG
from ssq_core import (
    RED_BALLS,
    parse_red_balls,
    validate_ball_scores,
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
)
from ssq_rule_models import RuleContext
from tqdm import tqdm

CandidateGenerationRequest = _selection_models.CandidateGenerationRequest
DuplexSelectionRequest = _selection_models.DuplexSelectionRequest
RedCandidateSelection = _selection_models.RedCandidateSelection
make_rejection_set = _anti_crowding.make_rejection_set
rejection_seed_for_issue = _anti_crowding.rejection_seed_for_issue
build_red_pool = _candidates.build_red_pool
count_actual_reds_by_rank_band = _candidates.count_actual_reds_by_rank_band
generate_candidates = _candidates.generate_candidates
generate_red_candidates = _candidates.generate_red_candidates
passes_red_filters = _candidates.passes_red_filters
select_recommendation_portfolio = _candidates.select_recommendation_portfolio
validate_candidate_generation_request = (
    _candidates.validate_candidate_generation_request
)


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
