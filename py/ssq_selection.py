"""Red-ball pool construction, candidate filtering, and ticket selection."""

import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import comb

from ssq_config import (
    DEFAULT_STRATEGY_CONFIG,
    RANDOM_SEED,
    REJECTION_SEED_MULTIPLIER,
    TOTAL_RED_COMBINATIONS,
)
from ssq_core import RED_BALLS, parse_issue, validate_ball_scores
from ssq_rules import (
    build_rank_center_scores,
    passes_red_filters,
    score_red_combination,
    select_recommendations,
)
from tqdm import tqdm

RANK_BAND_NAMES = ('high', 'middle', 'low', 'other')


def build_rank_bands(config=DEFAULT_STRATEGY_CONFIG):
    """Return score-rank bands matching the configured mixed pool."""
    total = len(RED_BALLS)
    middle_count = config.pool_size_red - config.high_count - config.low_count
    available_count = total - config.high_count - config.low_count
    middle_offset = max(0, (available_count - middle_count) // 2)
    middle_start = config.high_count + middle_offset + 1
    return {
        'high': range(1, config.high_count + 1),
        'middle': range(middle_start, middle_start + middle_count),
        'low': range(total - config.low_count + 1, total + 1),
    }


def build_rank_band_widths(config=DEFAULT_STRATEGY_CONFIG):
    bands = build_rank_bands(config)
    return {
        **{name: len(ranks) for name, ranks in bands.items()},
        'other': len(RED_BALLS) - sum(len(ranks) for ranks in bands.values()),
    }


def build_rank_band_labels(config=DEFAULT_STRATEGY_CONFIG):
    bands = build_rank_bands(config)

    def describe(title, ranks):
        return title if not ranks else f'{title}({ranks.start}-{ranks.stop - 1})'

    return {
        'high': describe('高端', bands['high']),
        'middle': describe('中段', bands['middle']),
        'low': describe('低端', bands['low']),
        'other': '其他',
    }


RANK_BANDS = build_rank_bands()
RANK_BAND_WIDTHS = build_rank_band_widths()


@dataclass(frozen=True)
class RedCandidateSelection:
    red_pool: tuple[int, ...]
    potential_combos: tuple[tuple[int, ...], ...]
    passed_combos: tuple[tuple[int, ...], ...]
    recommendations: tuple[tuple[int, ...], ...]


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


def generate_red_candidates(
    red_scores,
    context,
    rejection_set,
    config=DEFAULT_STRATEGY_CONFIG,
    mode='mixed',
    show_progress=False,
):
    """Run the shared red-ball selection pipeline for live runs and backtests."""
    red_pool = tuple(build_red_pool(red_scores, config=config, mode=mode))
    potential_combos = tuple(combinations(red_pool, 6))
    iterator = (
        tqdm(potential_combos, desc='规则过滤进度', ncols=80)
        if show_progress else potential_combos
    )
    passed_combos = tuple(
        combo for combo in iterator
        if passes_red_filters(combo, context, rejection_set)
    )
    recommendations = tuple(select_recommendations(
        passed_combos,
        red_scores,
        limit=config.recommendation_count,
        max_shared=config.max_shared_red_balls,
        context=context,
    ))
    return RedCandidateSelection(
        red_pool=red_pool,
        potential_combos=potential_combos,
        passed_combos=passed_combos,
        recommendations=recommendations,
    )


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


def find_best_7_red_combinations(
    passed_combos_tuples,
    red_pool,
    red_scores=None,
    last_draw_set=None,
    last_2_draw_set=None,
    context=None,
):
    """Rank every 7-red ticket by valid subticket coverage, then strategy score."""
    if not passed_combos_tuples:
        return []

    passed_combos_set = set(passed_combos_tuples)
    rank_center_scores = build_rank_center_scores(red_scores) if red_scores else None
    ranked = []
    seven_ball_combos = combinations(sorted(red_pool), 7)
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
        if red_scores:
            quality = sum(
                score_red_combination(
                    subticket,
                    red_scores,
                    last_draw_set,
                    last_2_draw_set,
                    rank_center_scores,
                    context,
                )
                for subticket in valid_subtickets
            ) / coverage
        ranked.append((seven_combo, coverage, quality))
    ranked.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [(combo, coverage) for combo, coverage, _ in ranked]
