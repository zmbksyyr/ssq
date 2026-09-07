"""Compatibility facade for red-ball selection modules."""

import ssq_anti_crowding as _anti_crowding
import ssq_candidates as _candidates
import ssq_duplex as _duplex
import ssq_selection_models as _selection_models
from ssq_config import DEFAULT_STRATEGY_CONFIG
from ssq_rank_bands import (
    build_rank_band_labels as _build_rank_band_labels,
)
from ssq_rank_bands import (
    build_rank_band_widths as _build_rank_band_widths,
)
from ssq_rank_bands import build_rank_bands as _build_rank_bands

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

build_combination_score_context = _duplex.build_combination_score_context
score_combination = _duplex.score_combination
find_best_7_red_combinations = _duplex.find_best_7_red_combinations
rank_duplex_candidates = _duplex.rank_duplex_candidates
validate_duplex_selection_request = _duplex.validate_duplex_selection_request


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
