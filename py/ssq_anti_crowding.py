"""Reproducible anti-crowding combination generation."""

import random

from ssq_config import (
    RANDOM_SEED,
    REJECTION_SEED_MULTIPLIER,
    TOTAL_RED_COMBINATIONS,
    normalize_integer_param,
)
from ssq_domain import RED_BALLS
from ssq_parsing import parse_issue


def make_rejection_set(size, rng=None):
    """Create a reproducible anti-crowding sample of six-red combinations."""
    size = normalize_integer_param('rejection size', size)
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
    base_seed = normalize_integer_param('base_seed', base_seed)
    return base_seed * REJECTION_SEED_MULTIPLIER + parse_issue(issue)
