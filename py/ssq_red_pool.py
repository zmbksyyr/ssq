"""Score-ranked red-ball pool construction and rank-band accounting."""

from collections import Counter

from ssq_config import DEFAULT_STRATEGY_CONFIG
from ssq_domain import RED_BALLS
from ssq_parsing import validate_ball_scores
from ssq_rank_bands import RANK_BAND_NAMES, build_rank_bands


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
