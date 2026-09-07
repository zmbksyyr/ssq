"""Red-ball pool construction, candidate filtering, and ticket selection."""

import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import comb

from ssq_config import (
    DEFAULT_STRATEGY_CONFIG,
    RANDOM_SEED,
    RED_HIGH_COUNT,
    RED_LOW_COUNT,
    REJECTION_SEED_MULTIPLIER,
    TOTAL_RED_COMBINATIONS,
)
from ssq_core import parse_issue
from ssq_rules import (
    build_rank_center_scores,
    passes_red_filters,
    score_red_combination,
    select_recommendations,
)
from tqdm import tqdm

RANK_BANDS = {
    'high': range(1, RED_HIGH_COUNT + 1),
    'middle': range(13, 22),
    'low': range(34 - RED_LOW_COUNT, 34),
}
RANK_BAND_WIDTHS = {
    **{name: len(ranks) for name, ranks in RANK_BANDS.items()},
    'other': 33 - sum(len(ranks) for ranks in RANK_BANDS.values()),
}


@dataclass(frozen=True)
class RedCandidateSelection:
    red_pool: tuple[int, ...]
    potential_combos: tuple[tuple[int, ...], ...]
    passed_combos: tuple[tuple[int, ...], ...]
    recommendations: tuple[tuple[int, ...], ...]


def count_actual_reds_by_rank_band(red_scores, actual_reds):
    """Count actual red balls by their model-score rank band."""
    ranked = [
        ball for ball, _ in sorted(
            red_scores.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    rank_by_ball = {ball: rank for rank, ball in enumerate(ranked, 1)}
    counts = Counter({'high': 0, 'middle': 0, 'low': 0, 'other': 0})
    for ball in actual_reds:
        rank = rank_by_ball[ball]
        band = next(
            (name for name, ranks in RANK_BANDS.items() if rank in ranks),
            'other',
        )
        counts[band] += 1
    return counts


def build_red_pool(red_scores, config=DEFAULT_STRATEGY_CONFIG, mode='mixed'):
    """Build a red pool from one score band or a high/middle/low mixture."""
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

    high = ranked[:config.high_count]
    low = ranked[-config.low_count:] if config.low_count else []
    middle_count = max(0, config.pool_size_red - len(high) - len(low))
    available_start = config.high_count
    available_end = (
        len(ranked) - config.low_count if config.low_count else len(ranked)
    )
    middle_start = available_start + max(
        0,
        (available_end - available_start - middle_count) // 2,
    )
    middle = ranked[middle_start:middle_start + middle_count]
    return sorted(dict.fromkeys(high + middle + low))


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
        context.last_draw,
        context.previous_draw,
        limit=config.recommendation_count,
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
        rejection_set.add(tuple(sorted(rng.sample(range(1, 34), 6))))
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
        coverage = sum(subticket in passed_combos_set for subticket in subtickets)
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
                )
                for subticket in subtickets
            ) / len(subtickets)
        ranked.append((seven_combo, coverage, quality))
    ranked.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [(combo, coverage) for combo, coverage, _ in ranked]
