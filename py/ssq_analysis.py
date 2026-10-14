"""Merged module: analysis (generated from 65 modules)."""

import argparse
import hashlib
import json
import os
import platform
import random
import select
import sys
import tempfile
import time

try:  # Windows-only; get_user_input_with_timeout() falls back to select()
    import msvcrt
except ImportError:  # pragma: no cover - exercised on Linux/macOS
    msvcrt = None

from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from importlib.metadata import version
from itertools import combinations, pairwise
from math import comb, isclose, isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm


# --- from ssq_report_models.py ---
@dataclass(frozen=True)
class AnalysisReportData:
    latest_issue: str
    target_issue: int
    generated_at: datetime
    params_loaded: bool
    params: dict
    config: Any
    rejection_seed: int
    pool_mode: str
    rank_band_widths: dict[str, int]
    rank_band_labels: dict[str, str]
    pipeline_stats: list[dict]
    selection: Any
    recommended_blues: list[int]
    best_7_reds: list
    runtime_versions: dict[str, str] = field(default_factory=dict)
    history_sha256: str = ''
    model_features: tuple[str, ...] = ()
    model_training_params: Mapping[str, Any] = field(default_factory=dict)


# --- from ssq_domain.py ---
BLUE_MAX = 16


# --- from ssq_domain.py ---
BLUE_MIN = 1


# --- from ssq_domain.py ---
BLUE_BALLS = tuple(range(BLUE_MIN, BLUE_MAX + 1))


# --- from ssq_config_models.py ---
MAX_SHARED_RED_BALLS = 4


# --- from ssq_config_models.py ---
NUM_BLUE_BALLS = 7


# --- from ssq_config_models.py ---
NUM_RECOMMENDATIONS = 10


# --- from ssq_config_models.py ---
POOL_SIZE_RED = 17


# --- from ssq_config_models.py ---
RANDOM_SEED = 42


# --- from ssq_domain.py ---
RED_MAX = 33


# --- from ssq_domain.py ---
RED_MIN = 1


# --- from ssq_domain.py ---
RED_BALLS = tuple(range(RED_MIN, RED_MAX + 1))


# --- from ssq_domain.py ---
RED_COUNT = 6


# --- from ssq_config_models.py ---
RED_HIGH_COUNT = 4


# --- from ssq_config_models.py ---
RED_LOW_COUNT = 4


# --- from ssq_config_models.py ---
REJECTION_LIB_SIZE = 500000


# --- from ssq_config_models.py ---
TOTAL_RED_COMBINATIONS = comb(len(RED_BALLS), RED_COUNT)


# --- from ssq_strategy_validation.py ---
def normalize_integer_param(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} 必须为整数')
    return int(value)


# --- from ssq_config_models.py ---
@dataclass(frozen=True)
class StrategyConfig:
    """Selection settings shared by backtests and live runs."""
    pool_size_red: int = POOL_SIZE_RED
    high_count: int = RED_HIGH_COUNT
    low_count: int = RED_LOW_COUNT
    blue_count: int = NUM_BLUE_BALLS
    recommendation_count: int = NUM_RECOMMENDATIONS
    max_shared_red_balls: int = MAX_SHARED_RED_BALLS
    rejection_lib_size: int = REJECTION_LIB_SIZE
    random_seed: int = RANDOM_SEED

    def __post_init__(self):
        integer_fields = ('pool_size_red', 'high_count', 'low_count', 'blue_count', 'recommendation_count', 'max_shared_red_balls', 'rejection_lib_size', 'random_seed')
        for name in integer_fields:
            object.__setattr__(self, name, normalize_integer_param(name, getattr(self, name)))
        if not RED_COUNT <= self.pool_size_red <= len(RED_BALLS):
            raise ValueError('pool_size_red must be between 6 and 33')
        if min(self.high_count, self.low_count) < 0:
            raise ValueError('high_count and low_count cannot be negative')
        if self.high_count + self.low_count > self.pool_size_red:
            raise ValueError('high_count and low_count exceed pool_size_red')
        if not 1 <= self.blue_count <= len(BLUE_BALLS):
            raise ValueError('blue_count must be between 1 and 16')
        if self.recommendation_count < 1:
            raise ValueError('recommendation_count must be positive')
        if not 0 <= self.max_shared_red_balls <= RED_COUNT:
            raise ValueError('max_shared_red_balls must be between 0 and 6')
        if not 0 <= self.rejection_lib_size <= TOTAL_RED_COMBINATIONS:
            raise ValueError(f'rejection_lib_size must be between 0 and {TOTAL_RED_COMBINATIONS}')


# --- from ssq_config_models.py ---
DEFAULT_STRATEGY_CONFIG = StrategyConfig()


# --- from ssq_rank_bands.py ---
def build_rank_bands(config=DEFAULT_STRATEGY_CONFIG):
    """Return score-rank bands matching the configured mixed pool."""
    total = len(RED_BALLS)
    middle_count = config.pool_size_red - config.high_count - config.low_count
    available_count = total - config.high_count - config.low_count
    middle_offset = max(0, (available_count - middle_count) // 2)
    middle_start = config.high_count + middle_offset + 1
    return {'high': range(1, config.high_count + 1), 'middle': range(middle_start, middle_start + middle_count), 'low': range(total - config.low_count + 1, total + 1)}


# --- from ssq_rank_bands.py ---
def build_rank_band_labels(config=DEFAULT_STRATEGY_CONFIG):
    bands = build_rank_bands(config)

    def describe(title, ranks):
        return title if not ranks else f'{title}({ranks.start}-{ranks.stop - 1})'
    return {'high': describe('高端', bands['high']), 'middle': describe('中段', bands['middle']), 'low': describe('低端', bands['low']), 'other': '其他'}


# --- from ssq_rank_bands.py ---
def build_rank_band_widths(config=DEFAULT_STRATEGY_CONFIG):
    bands = build_rank_bands(config)
    return {**{name: len(ranks) for name, ranks in bands.items()}, 'other': len(RED_BALLS) - sum(len(ranks) for ranks in bands.values())}


# --- from ssq_analysis_result.py ---
def build_analysis_report_data(history, evaluation, current, options, best_7_reds, generated_at, runtime_versions, model_training_params):
    """Map completed workflow stages into the report data contract."""
    config = options.strategy_config
    return AnalysisReportData(latest_issue=history.latest_issue, target_issue=history.target_issue, generated_at=generated_at, params_loaded=evaluation.loaded_params.loaded_from_file, params=evaluation.loaded_params.values, config=config, rejection_seed=current.rejection_seed, pool_mode=options.pool_mode, rank_band_widths=build_rank_band_widths(config), rank_band_labels=build_rank_band_labels(config), pipeline_stats=current.pipeline_stats, selection=current.candidate_selection, recommended_blues=current.recommended_blues, best_7_reds=best_7_reds, runtime_versions=runtime_versions, history_sha256=history.sha256, model_features=history.feature_columns, model_training_params=model_training_params)


# --- from ssq_anti_crowding.py ---
def make_rejection_set(size, rng=None):
    """Create a reproducible anti-crowding sample of six-red combinations."""
    size = normalize_integer_param('rejection size', size)
    if not 0 <= size <= TOTAL_RED_COMBINATIONS:
        raise ValueError(f'rejection size must be between 0 and {TOTAL_RED_COMBINATIONS}')
    rng = rng or random.Random(RANDOM_SEED)
    rejection_set = set()
    while len(rejection_set) < size:
        rejection_set.add(tuple(sorted(rng.sample(RED_BALLS, 6))))
    return rejection_set


# --- from ssq_config_models.py ---
REJECTION_SEED_MULTIPLIER = 1000000007


# --- from ssq_parsing.py ---
def parse_integer(value, field_name):
    if isinstance(value, bool):
        raise TypeError(f'{field_name}必须为整数')
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        if not isfinite(value) or not float(value).is_integer():
            raise ValueError(f'{field_name}必须为整数')
        return int(value)
    text = str(value).strip()
    if not text or not text.lstrip('+-').isdigit():
        raise ValueError(f'{field_name}必须为整数')
    return int(text)


# --- from ssq_parsing.py ---
def parse_issue(value):
    issue = parse_integer(value, '期号')
    year, sequence = divmod(issue, 1000)
    if year < 2003 or not 1 <= sequence <= 999:
        raise ValueError(f'无效期号: {value}')
    return issue


# --- from ssq_anti_crowding.py ---
def rejection_seed_for_issue(base_seed, issue):
    """Derive a reproducible anti-crowding seed that changes every issue."""
    base_seed = normalize_integer_param('base_seed', base_seed)
    return base_seed * REJECTION_SEED_MULTIPLIER + parse_issue(issue)


# --- from ssq_parsing.py ---
def validate_ball_scores(scores, candidates, label):
    """Return numeric scores after enforcing a complete ball domain."""
    expected = set(candidates)
    missing = sorted(expected - set(scores))
    extra = sorted(set(scores) - expected)
    if missing or extra:
        raise ValueError(f'{label}评分键不完整: 缺失 {missing}, 多余 {extra}')
    normalized = {}
    for ball in candidates:
        value = scores[ball]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f'{label} {ball} 的评分必须为数值')
        value = float(value)
        if not isfinite(value):
            raise ValueError(f'{label} {ball} 的评分必须为有限数值')
        normalized[ball] = value
    return normalized


# --- from ssq_red_pool.py ---
def build_red_pool(red_scores, config=DEFAULT_STRATEGY_CONFIG, mode='mixed'):
    """Build a red pool from one score band or a high/middle/low mixture."""
    red_scores = validate_ball_scores(red_scores, RED_BALLS, '红球')
    ranked = [ball for ball, _ in sorted(red_scores.items(), key=lambda item: (-item[1], item[0]))]
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
    selected_ranks = (*rank_bands['high'], *rank_bands['middle'], *rank_bands['low'])
    return sorted(ranked[rank - 1] for rank in selected_ranks)


# --- from ssq_rule_models.py ---
@dataclass(frozen=True)
class RuleContext:
    omission_values: Mapping[int, int] = field(default_factory=dict)
    recent_draws: Sequence[Collection[int]] = field(default_factory=tuple)
    last_draw: Collection[int] | None = None
    previous_draw: Collection[int] | None = None


# --- from ssq_rule_models.py ---
@dataclass(frozen=True)
class RuleDefinition:
    name: str
    hard: bool
    evaluator: Callable[[tuple[int, ...], RuleContext], bool]
    score_weight: float = 0.0
    scorer: Callable[[tuple[int, ...], RuleContext], float] | None = None


# --- from ssq_rule_functions.py ---
def calculate_ac_value(combo):
    standard_ac = len({abs(left - right) for left, right in combinations(combo, 2)})
    return standard_ac - 5


# --- from ssq_rule_functions.py ---
def filter_ac_value(combo):
    return 6 <= calculate_ac_value(combo) <= 10


# --- from ssq_rule_functions.py ---
def filter_all_cold(combo, omission_values):
    return not all(omission_values.get(ball, 0) > 15 for ball in combo)


# --- from ssq_rule_functions.py ---
def filter_big_small_ratio(combo):
    small_count = sum(ball <= 16 for ball in combo)
    return small_count not in (0, 1, 5, 6)


# --- from ssq_rule_functions.py ---
def filter_consecutive_numbers(combo):
    groups = 0
    max_length = 0
    current_length = 1
    for index in range(len(combo) - 1):
        if combo[index + 1] - combo[index] == 1:
            current_length += 1
        else:
            if current_length >= 2:
                groups += 1
                max_length = max(max_length, current_length)
            current_length = 1
    if current_length >= 2:
        groups += 1
        max_length = max(max_length, current_length)
    return not (groups >= 3 or max_length >= 4)


# --- from ssq_rule_functions.py ---
def filter_diagonal_consecutive(combo, last_draw, previous_draw):
    return not any(ball - 1 in last_draw and ball - 2 in previous_draw for ball in combo)


# --- from ssq_rule_functions.py ---
def filter_ending_digits(combo):
    counts = Counter(ball % 10 for ball in combo)
    return max(counts.values()) < 3 and len(counts) > 2


# --- from ssq_rule_functions.py ---
def filter_head_tail_range(combo):
    return combo[0] <= 10 and combo[-1] >= 25


# --- from ssq_rule_functions.py ---
def filter_highly_regular(combo):
    differences = {combo[index + 1] - combo[index] for index in range(len(combo) - 1)}
    return len(differences) > 1


# --- from ssq_rule_functions.py ---
def filter_modulo3_roads(combo):
    return len({ball % 3 for ball in combo}) == 3


# --- from ssq_rule_functions.py ---
def filter_odd_even_ratio(combo):
    even_count = sum(ball % 2 == 0 for ball in combo)
    return even_count not in (0, 1, 5, 6)


# --- from ssq_domain.py ---
PRIME_RED_BALLS = frozenset({2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31})


# --- from ssq_rule_functions.py ---
def is_prime(number):
    return number in PRIME_RED_BALLS


# --- from ssq_rule_functions.py ---
def filter_prime_composite_ratio(combo):
    prime_count = sum(is_prime(ball) for ball in combo)
    return prime_count not in (0, 1, 5, 6)


# --- from ssq_rule_functions.py ---
def filter_recent_overlap(combo, recent_draws):
    candidate = set(combo)
    return all(len(candidate & draw) < 4 for draw in recent_draws)


# --- from ssq_rule_functions.py ---
def filter_related_numbers(combo, last_draw):
    candidate = set(combo)
    repeats = candidate & last_draw
    adjacent_numbers = {number - 1 for number in last_draw} | {number + 1 for number in last_draw}
    return bool(repeats or candidate & adjacent_numbers)


# --- from ssq_rule_functions.py ---
def filter_span(combo):
    return combo[-1] - combo[0] >= 15


# --- from ssq_rule_functions.py ---
def filter_sum_of_tails(combo):
    return 15 <= sum(ball % 10 for ball in combo) <= 45


# --- from ssq_rule_functions.py ---
def filter_sum_value(combo):
    return 70 <= sum(combo) <= 160


# --- from ssq_rule_functions.py ---
def filter_zones(combo):
    all_in_small = all(ball <= 11 for ball in combo)
    all_in_medium = all(12 <= ball <= 22 for ball in combo)
    all_in_large = all(ball >= 23 for ball in combo)
    return not (all_in_small or all_in_medium or all_in_large)


# --- from ssq_rule_scoring.py ---
def score_big_small_balance(combo):
    return 1.0 - abs(sum(ball <= 16 for ball in combo) - 3) / 3


# --- from ssq_rule_scoring.py ---
def score_odd_even_balance(combo):
    return 1.0 - abs(sum(ball % 2 for ball in combo) - 3) / 3


# --- from ssq_rule_scoring.py ---
def score_prime_balance(combo):
    return 1.0 - abs(sum(is_prime(ball) for ball in combo) - 3) / 3


# --- from ssq_rule_scoring.py ---
def score_zone_balance(combo):
    counts = (sum(ball <= 11 for ball in combo), sum(12 <= ball <= 22 for ball in combo), sum(ball >= 23 for ball in combo))
    return 1.0 - (max(counts) - min(counts)) / 6


# --- from ssq_rule_definitions.py ---
RED_RULES = (RuleDefinition('highly_regular', True, lambda c, _: filter_highly_regular(c)), RuleDefinition('sum_value', True, lambda c, _: filter_sum_value(c)), RuleDefinition('span', True, lambda c, _: filter_span(c)), RuleDefinition('consecutive_numbers', True, lambda c, _: filter_consecutive_numbers(c)), RuleDefinition('zones', True, lambda c, _: filter_zones(c), 0.1, lambda c, _: score_zone_balance(c)), RuleDefinition('ac_value', False, lambda c, _: filter_ac_value(c), 0.05, lambda c, _: float(filter_ac_value(c))), RuleDefinition('prime_composite_ratio', False, lambda c, _: filter_prime_composite_ratio(c), 0.08, lambda c, _: score_prime_balance(c)), RuleDefinition('big_small_ratio', False, lambda c, _: filter_big_small_ratio(c), 0.08, lambda c, _: score_big_small_balance(c)), RuleDefinition('recent_overlap', True, lambda c, context: filter_recent_overlap(c, context.recent_draws)), RuleDefinition('all_cold', True, lambda c, context: filter_all_cold(c, context.omission_values)), RuleDefinition('odd_even_ratio', False, lambda c, _: filter_odd_even_ratio(c), 0.1, lambda c, _: score_odd_even_balance(c)), RuleDefinition('modulo3_roads', False, lambda c, _: filter_modulo3_roads(c), 0.04, lambda c, _: float(filter_modulo3_roads(c))), RuleDefinition('ending_digits', True, lambda c, _: filter_ending_digits(c)), RuleDefinition('head_tail_range', False, lambda c, _: filter_head_tail_range(c), 0.03, lambda c, _: float(filter_head_tail_range(c))), RuleDefinition('sum_of_tails', True, lambda c, _: filter_sum_of_tails(c)), RuleDefinition('related_numbers', True, lambda c, context: filter_related_numbers(c, context.last_draw or frozenset())), RuleDefinition('diagonal_consecutive', False, lambda c, context: filter_diagonal_consecutive(c, context.last_draw or (), context.previous_draw or ()), 0.02, lambda c, context: 1.0 if context.last_draw is None or context.previous_draw is None else float(filter_diagonal_consecutive(c, context.last_draw, context.previous_draw))))


# --- from entry.py ---
def _impl_passes_red_filters(combo, context, rule_definitions, rejection_set=None):
    passes_rules = all(not rule.hard or rule.evaluator(combo, context) for rule in rule_definitions)
    return passes_rules and (rejection_set is None or combo not in rejection_set)


# --- from entry.py ---
def passes_red_filters(combo, context, rejection_set=None):
    return _impl_passes_red_filters(combo, context, RED_RULES, rejection_set)


# --- from ssq_rule_models.py ---
@dataclass(frozen=True)
class CombinationScoreContext:
    red_scores: Mapping[int, float]
    rank_center_scores: Mapping[int, float]
    rule_context: RuleContext


# --- from ssq_combination_scoring.py ---
def build_rank_center_scores(red_scores):
    ranked = sorted(red_scores, key=lambda ball: (-red_scores[ball], ball))
    if len(ranked) <= 1:
        return {ball: 1.0 for ball in ranked}
    center = (len(ranked) - 1) / 2
    return {ball: 1.0 - abs(index - center) / center for index, ball in enumerate(ranked)}


# --- from ssq_combination_scoring.py ---
def build_combination_score_context(red_scores, context=None, rank_center_scores=None):
    """Build reusable inputs for ranking multiple red-ball combinations."""
    return CombinationScoreContext(red_scores=red_scores, rank_center_scores=rank_center_scores or build_rank_center_scores(red_scores), rule_context=context or RuleContext())


# --- from ssq_rule_definitions.py ---
COMBINATION_SIGNAL_WEIGHT = 0.5


# --- from ssq_combination_scoring.py ---
def score_rank_center_preference(combo, red_scores, rank_center_scores=None):
    rank_scores = rank_center_scores or build_rank_center_scores(red_scores)
    return sum(rank_scores.get(ball, 0.0) for ball in combo) / len(combo)


# --- from ssq_combination_scoring.py ---
def score_combination(combo, scoring_context, rule_definitions: Sequence[RuleDefinition] | None=None):
    """Score one combination from a precomputed ranking context."""
    rule_definitions = rule_definitions if rule_definitions is not None else RED_RULES
    if not isinstance(scoring_context, CombinationScoreContext):
        raise TypeError('scoring_context must be a CombinationScoreContext')
    signal = score_rank_center_preference(combo, scoring_context.red_scores, scoring_context.rank_center_scores)
    rule_score = sum(rule.score_weight * rule.scorer(combo, scoring_context.rule_context) for rule in rule_definitions if rule.scorer is not None)
    return COMBINATION_SIGNAL_WEIGHT * signal + rule_score


# --- from ssq_rule_models.py ---
@dataclass(frozen=True)
class RecommendationRequest:
    passed_combos: Iterable[tuple[int, ...]]
    red_scores: Mapping[int, float]
    context: RuleContext
    limit: int = NUM_RECOMMENDATIONS
    max_shared: int = MAX_SHARED_RED_BALLS


# --- from ssq_parsing.py ---
def parse_red_balls(value, expected_count=RED_COUNT):
    if isinstance(value, str):
        numbers = [parse_integer(part, '红球号码') for part in value.split(',')]
    else:
        numbers = [parse_integer(part, '红球号码') for part in value]
    if len(numbers) != expected_count:
        raise ValueError(f'红球数量必须为 {expected_count}，实际为 {len(numbers)}')
    if len(set(numbers)) != expected_count:
        raise ValueError('红球号码不能重复')
    if any(number < RED_MIN or number > RED_MAX for number in numbers):
        raise ValueError(f'红球号码必须在 {RED_MIN} 到 {RED_MAX} 之间')
    return sorted(numbers)


# --- from ssq_recommendation_validation.py ---
def validate_recommendation_request(request):
    """Validate and materialize recommendation inputs at the API boundary."""
    if not isinstance(request, RecommendationRequest):
        raise TypeError('request must be a RecommendationRequest')
    if not isinstance(request.context, RuleContext):
        raise TypeError('context must be a RuleContext')
    limit = normalize_integer_param('limit', request.limit)
    max_shared = normalize_integer_param('max_shared', request.max_shared)
    if limit < 0:
        raise ValueError('limit cannot be negative')
    if not 0 <= max_shared <= 6:
        raise ValueError('max_shared must be between 0 and 6')
    if not isinstance(request.red_scores, Mapping):
        raise TypeError('red_scores must be a mapping')
    if isinstance(request.passed_combos, (str, bytes)):
        raise TypeError('passed_combos must be an iterable of combinations')
    try:
        passed_combos = tuple(tuple(parse_red_balls(combo)) for combo in request.passed_combos)
    except TypeError as exc:
        raise TypeError('passed_combos must be an iterable of combinations') from exc
    if len(passed_combos) != len(set(passed_combos)):
        raise ValueError('passed_combos cannot contain duplicates')
    invalid_balls = [ball for ball in request.red_scores if ball not in RED_BALLS]
    if invalid_balls:
        raise ValueError(f'red_scores contains invalid balls: {invalid_balls}')
    required_balls = {ball for combo in passed_combos for ball in combo}
    missing_balls = sorted(required_balls - set(request.red_scores))
    if missing_balls:
        raise ValueError(f'red_scores is missing candidate balls: {missing_balls}')
    red_scores = {}
    for ball, value in request.red_scores.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f'red ball {ball} score must be numeric')
        value = float(value)
        if not isfinite(value):
            raise ValueError(f'red ball {ball} score must be finite')
        red_scores[ball] = value
    return (passed_combos, red_scores, limit, max_shared)


# --- from ssq_recommendation_portfolio.py ---
def select_recommendation_portfolio(request, rule_definitions: Sequence[RuleDefinition]=RED_RULES):
    """Select a diverse portfolio from ranked valid combinations."""
    passed_combos, red_scores, limit, max_shared = validate_recommendation_request(request)
    if limit == 0:
        return []
    scoring_context = build_combination_score_context(red_scores, request.context)
    ranked = sorted(passed_combos, key=lambda combo: (-score_combination(combo, scoring_context, rule_definitions), combo))
    selected = []
    selected_sets = []
    for overlap_limit in range(max_shared, 7):
        for combo in ranked:
            if combo in selected:
                continue
            candidate = set(combo)
            if all(len(candidate & previous) <= overlap_limit for previous in selected_sets):
                selected.append(combo)
                selected_sets.append(candidate)
                if len(selected) == limit:
                    return selected
    return selected


# --- from ssq_selection_models.py ---
@dataclass(frozen=True)
class CandidateGenerationRequest:
    red_scores: Mapping[int, float]
    context: RuleContext
    rejection_set: Collection[tuple[int, ...]] | None
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG
    mode: str = 'mixed'
    show_progress: bool = False


# --- from ssq_config_models.py ---
RED_POOL_MODES = ('mixed', 'high', 'middle', 'low')


# --- from ssq_candidate_validation.py ---
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
    if request.rejection_set is not None and (not isinstance(request.rejection_set, Collection) or isinstance(request.rejection_set, (str, bytes))):
        raise TypeError('rejection_set must be a collection or None')


# --- from ssq_candidate_generation.py ---
@dataclass(frozen=True)
class CandidateGenerationDependencies:
    """Replaceable boundaries used while enumerating and selecting candidates."""
    validate_request: Callable = validate_candidate_generation_request
    build_pool: Callable = build_red_pool
    passes_filters: Callable = passes_red_filters
    select_portfolio: Callable = select_recommendation_portfolio
    progress: Callable = tqdm


# --- from ssq_selection_models.py ---
@dataclass(frozen=True)
class RedCandidateSelection:
    red_pool: tuple[int, ...]
    potential_combos: tuple[tuple[int, ...], ...]
    passed_combos: tuple[tuple[int, ...], ...]
    recommendations: tuple[tuple[int, ...], ...]


# --- from entry.py ---
def _impl_generate_candidates(request, dependencies=None):
    """Run the shared red-ball selection pipeline for live runs and backtests."""
    dependencies = dependencies or CandidateGenerationDependencies()
    dependencies.validate_request(request)
    red_pool = tuple(dependencies.build_pool(request.red_scores, config=request.config, mode=request.mode))
    potential_combos = tuple(combinations(red_pool, 6))
    iterator = dependencies.progress(potential_combos, desc='规则过滤进度', ncols=80) if request.show_progress else potential_combos
    passed_combos = tuple(combo for combo in iterator if dependencies.passes_filters(combo, request.context, request.rejection_set))
    recommendations = tuple(dependencies.select_portfolio(RecommendationRequest(passed_combos=passed_combos, red_scores=request.red_scores, context=request.context, limit=request.config.recommendation_count, max_shared=request.config.max_shared_red_balls)))
    return RedCandidateSelection(red_pool=red_pool, potential_combos=potential_combos, passed_combos=passed_combos, recommendations=recommendations)


# --- from entry.py ---
def generate_candidates(request):
    """Run candidate generation through patchable legacy boundaries."""
    dependencies = CandidateGenerationDependencies(validate_request=validate_candidate_generation_request, build_pool=build_red_pool, passes_filters=passes_red_filters, select_portfolio=select_recommendation_portfolio, progress=tqdm)
    return _impl_generate_candidates(request, dependencies)


# --- from ssq_domain.py ---
PRIZE_RULES = {(6, 1): 5000000, (6, 0): 100000, (5, 1): 3000, (5, 0): 200, (4, 1): 200, (4, 0): 10, (3, 1): 10, (2, 1): 5, (1, 1): 5, (0, 1): 5}


# --- from ssq_rank_bands.py ---
RANK_BAND_WIDTHS = build_rank_band_widths()


# --- from ssq_rank_bands.py ---
RANK_BAND_NAMES = ('high', 'middle', 'low', 'other')


# --- from ssq_red_pool.py ---
def count_actual_reds_by_rank_band(red_scores, actual_reds, config=DEFAULT_STRATEGY_CONFIG):
    """Count actual red balls by their model-score rank band."""
    red_scores = validate_ball_scores(red_scores, RED_BALLS, '红球')
    rank_bands = build_rank_bands(config)
    ranked = [ball for ball, _ in sorted(red_scores.items(), key=lambda item: (-item[1], item[0]))]
    rank_by_ball = {ball: rank for rank, ball in enumerate(ranked, 1)}
    counts = Counter({name: 0 for name in RANK_BAND_NAMES})
    for ball in actual_reds:
        rank = rank_by_ball[ball]
        band = next((name for name, ranks in rank_bands.items() if rank in ranks), 'other')
        counts[band] += 1
    return counts


# --- from ssq_score_signals.py ---
def get_omission(df):
    """Calculate how many draws each red ball has been absent."""
    total_draws = len(df)
    last_positions = {}
    for position, draw in enumerate(df['红球']):
        for ball in draw:
            last_positions[ball] = position
    return {ball: total_draws - last_positions[ball] - 1 if ball in last_positions else total_draws for ball in RED_BALLS}


# --- from ssq_rule_auditing.py ---
def historical_rule_context(full_df, index):
    """Build rule inputs using only draws before the audited issue."""
    history = full_df.iloc[:index]
    recent = [set(draw) for draw in history.iloc[-10:]['红球']]
    return RuleContext(omission_values=get_omission(history), recent_draws=recent, last_draw=recent[-1], previous_draw=recent[-2])


# --- from ssq_score_adjustments.py ---
def apply_red_score_adjustments(red_scores, df_history, params):
    """Apply the hot, cold, and previous-draw bonuses."""
    adjusted = dict(red_scores)
    hot_lookback = int(params.get('hot_lookback', 0))
    hot_threshold = int(params.get('hot_threshold', 0))
    if hot_lookback > 0 and hot_threshold > 0:
        recent = df_history.tail(hot_lookback)['红球']
        hot_counts = Counter(ball for draw in recent for ball in draw)
        for ball, count in hot_counts.items():
            if count >= hot_threshold:
                adjusted[ball] *= params.get('hot_bonus', 1.0)
    cold_lookback = int(params.get('cold_lookback', 0))
    if cold_lookback > 0:
        recent_numbers = {ball for draw in df_history.tail(cold_lookback)['红球'] for ball in draw}
        for ball in set(RED_BALLS) - recent_numbers:
            adjusted[ball] *= params.get('cold_bonus', 1.0)
    if not df_history.empty:
        for ball in df_history.iloc[-1]['红球']:
            adjusted[ball] *= params.get('repeat_bonus', 1.0)
    return adjusted


# --- from ssq_score_signals.py ---
def get_weighted_frequency(series, decay_factor):
    """Calculate frequency with exponentially greater weight on recent draws."""
    draw_count = len(series)
    weights = np.array([decay_factor ** (draw_count - index - 1) for index in range(draw_count)])
    weighted_counts = {}
    for index, numbers in enumerate(series):
        for ball in numbers:
            weighted_counts[ball] = weighted_counts.get(ball, 0) + weights[index]
    return pd.Series(weighted_counts)


# --- from ssq_model_validation.py ---
def predict_positive_probability(model, features):
    """Return P(class=1), including correct behavior for one-class models."""
    classes = np.asarray(model.classes_)
    if len(classes) == 1:
        if classes[0] not in (0, 1):
            raise ValueError(f'单类别模型只能包含类别 0 或 1: {classes.tolist()}')
        return np.full(len(features), float(classes[0] == 1))
    positive_columns = np.flatnonzero(classes == 1)
    if len(positive_columns) != 1:
        raise ValueError(f'模型类别缺少唯一正类 1: {classes.tolist()}')
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    expected_shape = (len(features), len(classes))
    if probabilities.shape != expected_shape:
        raise ValueError(f'模型概率矩阵形状应为 {expected_shape}, 实际为 {probabilities.shape}')
    positive = probabilities[:, int(positive_columns[0])]
    if not np.isfinite(positive).all() or ((positive < 0) | (positive > 1)).any():
        raise ValueError('模型正类概率必须是 0 到 1 之间的有限数值')
    return positive


# --- from ssq_features.py ---
FEATURE_COLUMNS = ('red_sum', 'red_span', 'odd_count', 'blue_lag1', 'red_zone_small', 'red_zone_medium', 'red_zone_large', 'red_big_count', 'red_prime_count', 'red_sum_tail', 'red_consecutive_groups', 'red_ac_value', 'red_tail_uniques', 'red_sum_lag1', 'odd_count_lag1', 'red_sum_ma5', 'odd_count_ma5', 'blue_ma5')


# --- from ssq_features.py ---
def validate_feature_columns(frame, feature_columns=FEATURE_COLUMNS):
    """Reject missing, duplicate, or undeclared model feature names."""
    columns = tuple(feature_columns)
    if len(columns) != len(set(columns)):
        raise ValueError('模型特征列不能重复')
    missing = sorted(set(FEATURE_COLUMNS) - set(columns))
    undeclared = sorted(set(columns) - set(FEATURE_COLUMNS))
    if missing:
        raise ValueError(f'缺少声明的模型特征列: {missing}')
    if undeclared:
        raise ValueError(f'包含未声明的模型特征列: {undeclared}')
    if columns != FEATURE_COLUMNS:
        raise ValueError('模型特征列顺序与声明不一致')
    missing_from_frame = sorted(set(columns) - set(frame.columns))
    if missing_from_frame:
        raise ValueError(f'模型数据缺少特征列: {missing_from_frame}')
    return columns


# --- from ssq_model_validation.py ---
def validate_model_sets(red_models, blue_models):
    """Require exactly one trained model for every red and blue ball."""
    missing_red = sorted(set(RED_BALLS) - set(red_models))
    missing_blue = sorted(set(BLUE_BALLS) - set(blue_models))
    extra_red = sorted(set(red_models) - set(RED_BALLS))
    extra_blue = sorted(set(blue_models) - set(BLUE_BALLS))
    if missing_red or missing_blue or extra_red or extra_blue:
        raise ValueError(f'模型集合与号码范围不一致: 红球缺失 {missing_red}, 红球多余 {extra_red}, 蓝球缺失 {missing_blue}, 蓝球多余 {extra_blue}')


# --- from ssq_strategy_validation.py ---
DEFAULT_PARAMS = {'decay_factor': 0.999, 'weight_freq': 0.4, 'weight_omission': 0.5, 'weight_ml': 0.1, 'hot_lookback': 10, 'hot_threshold': 2, 'hot_bonus': 1.2, 'cold_lookback': 30, 'cold_bonus': 1.05, 'repeat_bonus': 1.15, 'weight_blue_freq': 0.6, 'weight_blue_ml': 0.4}


# --- from ssq_strategy_validation.py ---
FLOAT_PARAM_NAMES = ('decay_factor', 'weight_freq', 'weight_omission', 'weight_ml', 'hot_bonus', 'cold_bonus', 'repeat_bonus', 'weight_blue_freq', 'weight_blue_ml')


# --- from ssq_strategy_validation.py ---
INTEGER_PARAM_NAMES = ('hot_lookback', 'hot_threshold', 'cold_lookback')


# --- from ssq_strategy_validation.py ---
def normalize_float_param(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f'{name} 必须为数字')
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f'{name} 必须为有限数值')
    return normalized


# --- from ssq_strategy_validation.py ---
def validate_param_ranges(params):
    if not 0 < params['decay_factor'] <= 1:
        raise ValueError('decay_factor 必须在 (0, 1] 范围内')
    for name in INTEGER_PARAM_NAMES:
        if params[name] < 0:
            raise ValueError(f'{name} 不能为负数')
    for name in ('hot_bonus', 'cold_bonus', 'repeat_bonus'):
        if params[name] <= 0:
            raise ValueError(f'{name} 必须大于 0')


# --- from ssq_strategy_validation.py ---
def validate_weight_group(params, names):
    values = [params[name] for name in names]
    if any(value < 0 for value in values) or not np.isclose(sum(values), 1.0):
        raise ValueError(f"权重 {', '.join(names)} 必须非负且总和为 1")


# --- from ssq_strategy_validation.py ---
def validate_strategy_params(params):
    if not isinstance(params, dict):
        raise TypeError('strategy params must be a JSON object')
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise ValueError(f"未知策略参数: {', '.join(unknown)}")
    merged = {**DEFAULT_PARAMS, **params}
    for name in INTEGER_PARAM_NAMES:
        merged[name] = normalize_integer_param(name, merged[name])
    for name in FLOAT_PARAM_NAMES:
        merged[name] = normalize_float_param(name, merged[name])
    for group in (('weight_freq', 'weight_omission', 'weight_ml'), ('weight_blue_freq', 'weight_blue_ml')):
        validate_weight_group(merged, group)
    validate_param_ranges(merged)
    return merged


# --- from ssq_ball_scoring.py ---
@dataclass(frozen=True)
class BallScoringDependencies:
    """Replaceable calculations used by score fusion."""
    validate_params: Callable[..., dict] = validate_strategy_params
    validate_models: Callable[..., None] = validate_model_sets
    validate_features: Callable[..., tuple] = validate_feature_columns
    predict_probability: Callable[..., Any] = predict_positive_probability
    weighted_frequency: Callable[..., Any] = get_weighted_frequency
    omission: Callable[..., dict] = get_omission
    adjust_red_scores: Callable[..., dict] = apply_red_score_adjustments
    validate_scores: Callable[..., dict] = validate_ball_scores


# --- from ssq_ball_scoring.py ---
def run_strategy_and_get_scores(df_history, params, ml_models_red, ml_models_blue, feature_columns, dependencies=None):
    """Combine frequency, omission, and model signals into ball scores."""
    dependencies = dependencies or BallScoringDependencies()
    params = dependencies.validate_params(params)
    dependencies.validate_models(ml_models_red, ml_models_blue)
    feature_columns = dependencies.validate_features(df_history, feature_columns)
    if df_history.empty:
        raise ValueError('评分历史数据不能为空')
    last_features = df_history.iloc[[-1]][list(feature_columns)].copy()
    for column in last_features.columns:
        if last_features[column].isnull().any():
            last_features[column] = last_features[column].fillna(df_history[column].mean())
    try:
        feature_values = last_features.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('最新一期模型特征必须为数值') from exc
    if not np.isfinite(feature_values).all():
        raise ValueError('最新一期模型特征包含无法填补的非有限值')
    red_weighted_freq = dependencies.weighted_frequency(df_history['红球'], params['decay_factor'])
    red_omission = dependencies.omission(df_history)
    red_ml_probs = {ball: dependencies.predict_probability(ml_models_red[ball], last_features)[0] for ball in RED_BALLS}
    max_red_freq = red_weighted_freq.max() or 1
    max_red_omission = max(red_omission.values()) or 1
    red_scores = {ball: red_weighted_freq.get(ball, 0) / max_red_freq * params['weight_freq'] + red_omission.get(ball, 0) / max_red_omission * params['weight_omission'] + red_ml_probs[ball] * params['weight_ml'] for ball in RED_BALLS}
    red_scores = dependencies.adjust_red_scores(red_scores, df_history, params)
    red_scores = dependencies.validate_scores(red_scores, RED_BALLS, '红球')
    blue_weighted_freq = dependencies.weighted_frequency(df_history['蓝球'].apply(lambda ball: [ball]), params['decay_factor'])
    blue_ml_probs = {ball: dependencies.predict_probability(ml_models_blue[ball], last_features)[0] for ball in BLUE_BALLS}
    max_blue_freq = blue_weighted_freq.max() or 1
    blue_scores = {ball: blue_weighted_freq.get(ball, 0) / max_blue_freq * params['weight_blue_freq'] + blue_ml_probs[ball] * params['weight_blue_ml'] for ball in BLUE_BALLS}
    blue_scores = dependencies.validate_scores(blue_scores, BLUE_BALLS, '蓝球')
    return (red_scores, blue_scores)


# --- from ssq_model_contracts.py ---
@dataclass(frozen=True)
class BallModelSpec:
    candidates: Sequence[int]
    outcome_column: str
    contains_candidate: Callable[[object, int], bool]
    description: str | None = None


# --- from ssq_model_training.py ---
@dataclass(frozen=True)
class PredictionTrainingDependencies:
    validate_features: Callable
    train_spec: Callable


# --- from ssq_model_contracts.py ---
MODEL_TRAINING_PARAMS = MappingProxyType({'random_state': 42, 'deterministic': True, 'force_col_wise': True, 'verbose': -1})


# --- from ssq_model_training.py ---
@dataclass(frozen=True)
class ModelFitDependencies:
    classifier_factory: Callable
    progress_factory: Callable


# --- from ssq_model_training.py ---
def default_fit_dependencies():
    return ModelFitDependencies(classifier_factory=lgb.LGBMClassifier, progress_factory=tqdm)


# --- from ssq_model_training.py ---
def train_models_for_spec(training_df, feature_columns, spec, dependencies=None):
    """Train one binary next-draw model per candidate."""
    if not isinstance(spec, BallModelSpec):
        raise TypeError('spec 必须为 BallModelSpec')
    dependencies = dependencies or default_fit_dependencies()
    models = {}
    iterator = dependencies.progress_factory(spec.candidates, desc=spec.description, ncols=80) if spec.description else spec.candidates
    features = training_df[list(feature_columns)]
    for candidate in iterator:
        target = training_df[spec.outcome_column].apply(lambda outcome, current=candidate: int(spec.contains_candidate(outcome, current))).shift(-1)
        valid_rows = target.notna() & features.notna().all(axis=1)
        if not valid_rows.any():
            continue
        model = dependencies.classifier_factory(**MODEL_TRAINING_PARAMS)
        model.fit(features.loc[valid_rows], target.loc[valid_rows])
        models[candidate] = model
    return models


# --- from ssq_model_training.py ---
def default_prediction_dependencies():
    return PredictionTrainingDependencies(validate_features=validate_feature_columns, train_spec=train_models_for_spec)


# --- from ssq_model_training.py ---
def train_prediction_models(training_df, feature_columns, show_progress=False, dependencies=None):
    """Train the complete red and blue model sets."""
    dependencies = dependencies or default_prediction_dependencies()
    feature_columns = dependencies.validate_features(training_df, feature_columns)
    red_models = dependencies.train_spec(training_df, feature_columns, BallModelSpec(candidates=RED_BALLS, outcome_column='红球', contains_candidate=lambda draw, ball: ball in draw, description='训练红球模型' if show_progress else None))
    blue_models = dependencies.train_spec(training_df, feature_columns, BallModelSpec(candidates=BLUE_BALLS, outcome_column='蓝球', contains_candidate=lambda drawn, ball: drawn == ball, description='训练蓝球模型' if show_progress else None))
    return (red_models, blue_models)


# --- from ssq_domain.py ---
PRIZE_NAMES = {(6, 1): '一等奖', (6, 0): '二等奖', (5, 1): '三等奖', (5, 0): '四等奖', (4, 1): '四等奖', (4, 0): '五等奖', (3, 1): '五等奖', (2, 1): '六等奖', (1, 1): '六等奖', (0, 1): '六等奖'}


# --- from ssq_config_models.py ---
RULE_AUDIT_PERIODS = 200


# --- from entry.py ---
def _impl_explain_filter_failures(combo, context, rule_definitions, rejection_set=None):
    """Return every configured rule and anti-crowding check that fails."""
    failures = [rule.name for rule in rule_definitions if not rule.evaluator(combo, context)]
    if rejection_set is not None and combo in rejection_set:
        failures.append('anti_crowding')
    return failures


# --- from entry.py ---
def explain_filter_failures(combo, context, rejection_set=None):
    return _impl_explain_filter_failures(combo, context, RED_RULES, rejection_set)


# --- from ssq_candidates.py ---
def generate_red_candidates(red_scores, context, rejection_set, config=DEFAULT_STRATEGY_CONFIG, mode='mixed', show_progress=False):
    """Compatibility wrapper for request-based candidate generation."""
    return generate_candidates(CandidateGenerationRequest(red_scores=red_scores, context=context, rejection_set=rejection_set, config=config, mode=mode, show_progress=show_progress))


# --- from ssq_cli.py ---
def build_argument_parser():
    parser = argparse.ArgumentParser(description='双色球策略分析与推荐')
    parser.add_argument('--rejection-size', type=int, default=REJECTION_LIB_SIZE, help='反撞号随机库大小，默认 %(default)s')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='随机种子，默认 %(default)s')
    parser.add_argument('--pool-mode', choices=RED_POOL_MODES, default='mixed', help='红球候选池模式，默认 %(default)s')
    parser.add_argument('--non-interactive', action='store_true', help='不等待键盘输入，适用于自动化运行')
    return parser


# --- from ssq_config_models.py ---
@dataclass(frozen=True)
class AnalyzerOptions:
    rejection_size: int = REJECTION_LIB_SIZE
    seed: int = RANDOM_SEED
    pool_mode: str = 'mixed'
    non_interactive: bool = False

    def __post_init__(self):
        integer_fields = ('rejection_size', 'seed')
        for name in integer_fields:
            object.__setattr__(self, name, normalize_integer_param(name, getattr(self, name)))
        if not isinstance(self.non_interactive, bool):
            raise TypeError('non_interactive 必须为布尔值')
        if not 0 <= self.rejection_size <= TOTAL_RED_COMBINATIONS:
            raise ValueError(f'rejection-size 必须在 0 到 {TOTAL_RED_COMBINATIONS} 之间')
        if self.pool_mode not in RED_POOL_MODES:
            raise ValueError(f'未知候选池模式: {self.pool_mode}')

    @property
    def strategy_config(self):
        return StrategyConfig(rejection_lib_size=self.rejection_size, random_seed=self.seed)


# --- from ssq_cli.py ---
def parse_cli_options(argv=None):
    parser = build_argument_parser()
    namespace = parser.parse_args(argv)
    try:
        return AnalyzerOptions(**vars(namespace))
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


# --- from ssq_combination_scoring.py ---
def score_red_combination(combo, red_scores, last_draw=None, previous_draw=None, rank_center_scores=None, context=None, rule_definitions=None):
    """Score a combination with individual compatibility-style arguments."""
    rule_definitions = rule_definitions if rule_definitions is not None else RED_RULES
    return score_combination(combo, build_combination_score_context(red_scores, context=context or RuleContext(last_draw=last_draw, previous_draw=previous_draw), rank_center_scores=rank_center_scores), rule_definitions)


# --- from ssq_config_models.py ---
INTERACTIVE_THRESHOLD = 100


# --- from ssq_config_models.py ---
COUNTDOWN_SECONDS = 10


# --- from ssq_config_models.py ---
@dataclass(frozen=True)
class LoadedStrategyParams:
    values: dict
    loaded_from_file: bool


# --- from ssq_console.py ---
def is_confirmation_input(value):
    """Return whether terminal input explicitly confirms the prompt."""
    if isinstance(value, bytes):
        value = value.decode(errors='ignore')
    return value.strip().lower() == 'y'


# --- from ssq_console.py ---
def get_user_input_with_timeout(timeout):
    """Wait up to ``timeout`` seconds and return whether the user confirmed."""
    prompt = f"\n发现大量高质量组合。输入 'y' 并回车可在 {timeout} 秒内查看全部，否则将仅输出随机推荐...\n"
    sys.stdout.write(prompt)
    sys.stdout.flush()
    confirmed = False
    if msvcrt is not None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if msvcrt.kbhit() and is_confirmation_input(msvcrt.getch()):
                confirmed = True
                break
            time.sleep(0.1)
    else:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            confirmed = is_confirmation_input(sys.stdin.readline())
    sys.stdout.write('\n倒计时结束。\n')
    sys.stdout.flush()
    return confirmed


# --- from ssq_console.py ---
def display_passed_combinations(passed_combos, non_interactive):
    """Display candidate combinations according to the existing CLI policy."""
    if 0 < len(passed_combos) < INTERACTIVE_THRESHOLD:
        print(f'\n通过检验的组合数量为 {len(passed_combos)} (低于{INTERACTIVE_THRESHOLD})，全部输出如下：')
        for index, combo in enumerate(passed_combos, 1):
            print(f"  组合 {index:>2}: {' '.join(f'{number:02d}' for number in combo)}")
    elif len(passed_combos) >= INTERACTIVE_THRESHOLD and (not non_interactive) and get_user_input_with_timeout(COUNTDOWN_SECONDS):
        print('\n根据您的确认，输出所有通过检验的组合：')
        for index, combo in enumerate(passed_combos, 1):
            print(f"  组合 {index:>3}: {' '.join(f'{number:02d}' for number in combo)}")


# --- from ssq_domain.py ---
DRAW_COLUMNS = ('期号', '日期', '红球', '蓝球')


# --- from ssq_domain.py ---
DRAW_WEEKDAYS = {1, 3, 6}


# --- from ssq_domain.py ---
LOCAL_TIMEZONE = ZoneInfo('Asia/Shanghai')


# --- from ssq_draw_schedule.py ---
def local_now():
    return datetime.now(LOCAL_TIMEZONE)


# --- from ssq_draw_schedule.py ---
def local_today():
    return local_now().date()


# --- from ssq_draw_data.py ---
def validate_draw_dates_not_future(frame, today=None):
    """Reject draw records dated after the current local calendar date."""
    if frame.empty:
        return
    today = today or local_today()
    latest_date = pd.to_datetime(frame['日期'], format='%Y-%m-%d', errors='raise').max().date()
    if latest_date > today:
        raise ValueError(f'开奖记录包含未来开奖日期: {latest_date}')


# --- from ssq_parsing.py ---
def parse_blue_ball(value):
    number = parse_integer(value, '蓝球号码')
    if number < BLUE_MIN or number > BLUE_MAX:
        raise ValueError(f'蓝球号码必须在 {BLUE_MIN} 到 {BLUE_MAX} 之间')
    return number


# --- from ssq_draw_data.py ---
def normalize_draw_frame(frame):
    """Return a validated draw table in chronological, typed form."""
    actual_columns = set(frame.columns)
    expected_columns = set(DRAW_COLUMNS)
    if actual_columns != expected_columns or len(frame.columns) != len(DRAW_COLUMNS):
        missing = sorted(expected_columns - actual_columns)
        unexpected = sorted(actual_columns - expected_columns)
        raise ValueError(f'字段不匹配: 缺少 {missing}, 多余 {unexpected}')
    normalized = frame.loc[:, list(DRAW_COLUMNS)].copy()
    normalized['期号'] = normalized['期号'].apply(parse_issue)
    if normalized['期号'].duplicated().any():
        duplicates = sorted(int(issue) for issue in normalized.loc[normalized['期号'].duplicated(keep=False), '期号'].unique())
        raise ValueError(f'存在重复期号: {duplicates}')
    parsed_dates = pd.to_datetime(normalized['日期'], format='%Y-%m-%d', errors='raise')
    normalized['日期'] = parsed_dates
    normalized['红球'] = normalized['红球'].apply(parse_red_balls)
    normalized['蓝球'] = normalized['蓝球'].apply(parse_blue_ball)
    normalized = normalized.sort_values('期号').reset_index(drop=True)
    if (normalized['期号'] // 1000 != normalized['日期'].dt.year).any():
        raise ValueError('期号年份与开奖日期不一致')
    if normalized['日期'].duplicated().any():
        raise ValueError('同一开奖日期不能对应多个期号')
    if not normalized['日期'].is_monotonic_increasing:
        raise ValueError('期号与开奖日期顺序不一致')
    if not normalized['日期'].dt.weekday.isin(DRAW_WEEKDAYS).all():
        raise ValueError('开奖日期必须为周二、周四或周日')
    issues = normalized['期号'].tolist()
    for previous, current in pairwise(issues):
        previous_year, previous_sequence = divmod(previous, 1000)
        current_year, current_sequence = divmod(current, 1000)
        continuous = current_year == previous_year and current_sequence == previous_sequence + 1 or (current_year == previous_year + 1 and current_sequence == 1)
        if not continuous:
            raise ValueError(f'开奖期号不连续: {previous} -> {current}')
    normalized['日期'] = normalized['日期'].dt.strftime('%Y-%m-%d')
    return normalized


# --- from ssq_draw_data.py ---
def serialize_draw_frame(frame):
    """Return a validated draw table using the stable on-disk CSV format."""
    serialized = normalize_draw_frame(frame)
    serialized['红球'] = serialized['红球'].apply(lambda numbers: ','.join(f'{number:02d}' for number in numbers))
    serialized['蓝球'] = serialized['蓝球'].apply(lambda number: f'{number:02d}')
    return serialized


# --- from ssq_draw_data.py ---
def fingerprint_draw_frame(frame):
    """Return a stable fingerprint of normalized, serialized draw history."""
    canonical_csv = serialize_draw_frame(frame).to_csv(index=False, lineterminator='\n')
    return hashlib.sha256(canonical_csv.encode('utf-8')).hexdigest()


# --- from ssq_draw_schedule.py ---
def infer_next_issue(current_issue, current_draw_date):
    """Infer the next regular draw issue, including the year boundary."""
    issue = parse_issue(current_issue)
    issue_year, _ = divmod(issue, 1000)
    if isinstance(current_draw_date, datetime):
        draw_date = current_draw_date.date()
    elif isinstance(current_draw_date, date):
        draw_date = current_draw_date
    else:
        draw_date = date.fromisoformat(str(current_draw_date))
    if draw_date.year != issue_year:
        raise ValueError(f'期号年份 {issue_year} 与开奖日期 {draw_date} 不一致')
    next_draw_date = draw_date + timedelta(days=1)
    while next_draw_date.weekday() not in DRAW_WEEKDAYS:
        next_draw_date += timedelta(days=1)
    if next_draw_date.year != issue_year:
        return next_draw_date.year * 1000 + 1
    return issue + 1


# --- from ssq_selection_models.py ---
@dataclass(frozen=True)
class DuplexSelectionRequest:
    passed_combos: Collection[tuple[int, ...]]
    red_pool: Collection[int]
    red_scores: Mapping[int, float] | None = None
    context: RuleContext = field(default_factory=RuleContext)


# --- from ssq_duplex.py ---
def validate_duplex_selection_request(request):
    """Validate and materialize duplex inputs before exhaustive ranking."""
    if not isinstance(request, DuplexSelectionRequest):
        raise TypeError('request must be a DuplexSelectionRequest')
    if not isinstance(request.context, RuleContext):
        raise TypeError('context must be a RuleContext')
    if not isinstance(request.red_pool, Collection) or isinstance(request.red_pool, (str, bytes)):
        raise TypeError('red_pool must be a collection')
    red_pool = tuple(parse_red_balls(request.red_pool, expected_count=len(request.red_pool)))
    if len(red_pool) < 7:
        raise ValueError('red_pool must contain at least 7 balls')
    if not isinstance(request.passed_combos, Collection) or isinstance(request.passed_combos, (str, bytes)):
        raise TypeError('passed_combos must be a collection')
    passed_combos = tuple(tuple(parse_red_balls(combo)) for combo in request.passed_combos)
    if len(passed_combos) != len(set(passed_combos)):
        raise ValueError('passed_combos cannot contain duplicates')
    if any(not set(combo).issubset(red_pool) for combo in passed_combos):
        raise ValueError('every passed combination must belong to red_pool')
    red_scores = validate_ball_scores(request.red_scores, RED_BALLS, 'red ball') if request.red_scores is not None else None
    return (passed_combos, red_pool, red_scores)


# --- from ssq_duplex.py ---
def rank_duplex_candidates(request):
    """Rank every 7-red ticket by valid subticket coverage, then strategy score."""
    passed_combos, red_pool, red_scores = validate_duplex_selection_request(request)
    if not passed_combos:
        return []
    passed_combos_set = set(passed_combos)
    scoring_context = build_combination_score_context(red_scores, request.context) if red_scores is not None else None
    ranked = []
    seven_ball_combos = combinations(red_pool, 7)
    for seven_combo in tqdm(seven_ball_combos, total=comb(len(red_pool), 7), desc='生成7红球大底', leave=False, ncols=80):
        valid_subtickets = [subticket for subticket in combinations(seven_combo, 6) if subticket in passed_combos_set]
        coverage = len(valid_subtickets)
        if not coverage:
            continue
        quality = 0.0
        if scoring_context:
            quality = sum(score_combination(subticket, scoring_context) for subticket in valid_subtickets) / coverage
        ranked.append((seven_combo, coverage, quality))
    ranked.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [(combo, coverage) for combo, coverage, _ in ranked]


# --- from ssq_duplex.py ---
def find_best_7_red_combinations(passed_combos_tuples, red_pool, red_scores=None, last_draw_set=None, last_2_draw_set=None, context=None):
    """Compatibility wrapper for request-based duplex selection."""
    return rank_duplex_candidates(DuplexSelectionRequest(passed_combos=passed_combos_tuples, red_pool=red_pool, red_scores=red_scores, context=context or RuleContext(last_draw=last_draw_set, previous_draw=last_2_draw_set)))


# --- from ssq_features.py ---
def _count_consecutive_groups(numbers):
    groups = 0
    in_group = False
    for current, following in pairwise(numbers):
        if following - current == 1:
            if not in_group:
                groups += 1
                in_group = True
        else:
            in_group = False
    return groups


# --- from ssq_features.py ---
def feature_engineer(df):
    """Return a copy of draw history with the declared model input features."""
    df = df.copy()
    df['red_sum'] = df['红球'].apply(sum)
    df['red_span'] = df['红球'].apply(lambda numbers: max(numbers) - min(numbers))
    df['odd_count'] = df['红球'].apply(lambda numbers: sum(number % 2 != 0 for number in numbers))
    df['blue_lag1'] = df['蓝球'].shift(1)
    df['red_zone_small'] = df['红球'].apply(lambda numbers: sum(1 <= number <= 11 for number in numbers))
    df['red_zone_medium'] = df['红球'].apply(lambda numbers: sum(12 <= number <= 22 for number in numbers))
    df['red_zone_large'] = df['红球'].apply(lambda numbers: sum(23 <= number <= 33 for number in numbers))
    df['red_big_count'] = df['红球'].apply(lambda numbers: sum(number > 16 for number in numbers))
    df['red_prime_count'] = df['红球'].apply(lambda numbers: sum(number in PRIME_RED_BALLS for number in numbers))
    df['red_sum_tail'] = df['red_sum'].apply(lambda value: value % 10)
    df['red_consecutive_groups'] = df['红球'].apply(_count_consecutive_groups)
    df['red_ac_value'] = df['红球'].apply(lambda numbers: len({abs(first - second) for first, second in combinations(numbers, 2)}))
    df['red_tail_uniques'] = df['红球'].apply(lambda numbers: len({number % 10 for number in numbers}))
    window_size = 5
    df['red_sum_lag1'] = df['red_sum'].shift(1)
    df['odd_count_lag1'] = df['odd_count'].shift(1)
    df['red_sum_ma5'] = df['red_sum'].shift(1).rolling(window=window_size).mean()
    df['odd_count_ma5'] = df['odd_count'].shift(1).rolling(window=window_size).mean()
    df['blue_ma5'] = df['蓝球'].shift(1).rolling(window=window_size).mean()
    return df


# --- from ssq_file_io.py ---
@contextmanager
def atomic_text_writer(path, encoding='utf-8'):
    """Yield an adjacent temporary text file and replace the target on success."""
    target = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding=encoding, newline='', dir=directory, prefix='.ssq-', suffix='.tmp', delete=False) as temporary_file:
            temporary_path = temporary_file.name
            yield temporary_file
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


# --- from ssq_file_io.py ---
def atomic_write_text(path, content, encoding='utf-8'):
    """Atomically replace a text file with the supplied content."""
    with atomic_text_writer(path, encoding=encoding) as temporary_file:
        temporary_file.write(content)


# --- from ssq_history_evaluation.py ---
@dataclass(frozen=True)
class HistoryEvaluationDependencies:
    load_params: Callable[..., Any]


# --- from ssq_strategy_params.py ---
def load_strategy_params(filepath):
    try:
        with open(filepath, encoding='utf-8') as handle:
            values = validate_strategy_params(json.load(handle))
    except FileNotFoundError:
        return LoadedStrategyParams(validate_strategy_params({}), False)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f'参数文件 {filepath} 无效: {exc}') from exc
    return LoadedStrategyParams(values, True)


# --- from ssq_history_evaluation.py ---
def default_evaluation_dependencies():
    return HistoryEvaluationDependencies(load_params=load_strategy_params)


# --- from ssq_workflow_models.py ---
@dataclass(frozen=True)
class HistoricalEvaluation:
    loaded_params: LoadedStrategyParams


# --- from ssq_paths.py ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# --- from ssq_paths.py ---
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# --- from ssq_paths.py ---
PARAMS_JSON_PATH = os.path.join(PROJECT_ROOT, 'best_params.json')


# --- from ssq_history_evaluation.py ---
def evaluate_history(history, options, params_path=PARAMS_JSON_PATH, dependencies=None):
    """Load the strategy parameters used for the target issue."""
    params_path = params_path if params_path is not None else PARAMS_JSON_PATH
    if dependencies is None:
        dependencies = default_evaluation_dependencies()
    try:
        loaded_params = dependencies.load_params(params_path)
    except ValueError as exc:
        raise SystemExit(f'错误: {exc}') from exc
    if not loaded_params.loaded_from_file:
        print(f'警告: 未找到参数文件 {params_path}，将使用内置的默认参数。')
    return HistoricalEvaluation(loaded_params=loaded_params)


# --- from ssq_history_preparation.py ---
@dataclass(frozen=True)
class HistoryPreparationDependencies:
    load_data: Callable[..., Any]
    fingerprint_history: Callable[..., str]
    engineer_features: Callable[..., pd.DataFrame]
    infer_target_issue: Callable[..., int]


# --- from ssq_history_preparation.py ---
def load_and_preprocess_data(filepath, normalize=None, validate_dates=None):
    """Load, validate, parse, and chronologically order draw history."""
    normalize = normalize or normalize_draw_frame
    validate_dates = validate_dates or validate_draw_dates_not_future
    try:
        frame = pd.read_csv(filepath, header=0)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        print(f"错误: 无法加载数据文件 '{filepath}': {exc}")
        return None
    try:
        normalized = normalize(frame)
        validate_dates(normalized)
        return normalized
    except (TypeError, ValueError) as exc:
        print(f"错误: 数据文件 '{filepath}' 校验失败: {exc}")
        return None


# --- from ssq_history_preparation.py ---
def default_history_dependencies():
    return HistoryPreparationDependencies(load_data=load_and_preprocess_data, fingerprint_history=fingerprint_draw_frame, engineer_features=feature_engineer, infer_target_issue=infer_next_issue)


# --- from ssq_paths.py ---
CSV_PATH = os.path.join(PROJECT_ROOT, 'shuangseqiu.csv')


# --- from ssq_workflow_models.py ---
@dataclass(frozen=True)
class PreparedHistory:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    latest_issue: str
    target_issue: int
    sha256: str


# --- from ssq_history_preparation.py ---
def prepare_history(filepath=CSV_PATH, dependencies=None):
    """Derive immutable analysis inputs from validated historical draws."""
    filepath = filepath if filepath is not None else CSV_PATH
    if dependencies is None:
        dependencies = default_history_dependencies()
    full_df = dependencies.load_data(filepath)
    if full_df is None or len(full_df) < 50:
        raise SystemExit('错误: 历史数据加载失败或数据量过少（至少需要50期），程序终止。')
    history_sha256 = dependencies.fingerprint_history(full_df)
    full_df = dependencies.engineer_features(full_df)
    latest_issue = str(full_df.iloc[-1]['期号'])
    try:
        target_issue = dependencies.infer_target_issue(latest_issue, full_df.iloc[-1]['日期'])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f'错误: 无法推导下一期期号: {exc}') from exc
    return PreparedHistory(frame=full_df, feature_columns=FEATURE_COLUMNS, latest_issue=latest_issue, target_issue=target_issue, sha256=history_sha256)


# --- from ssq_model_training.py ---
def train_ball_models(training_df, feature_columns, candidates, outcome_column, contains_candidate, description=None, dependencies=None):
    """Train a caller-defined family of per-ball binary models."""
    return train_models_for_spec(training_df, feature_columns, BallModelSpec(candidates=candidates, outcome_column=outcome_column, contains_candidate=contains_candidate, description=description), dependencies)


# --- from ssq_parsing.py ---
def parse_blue_balls(value):
    if isinstance(value, str):
        numbers = [parse_blue_ball(part.strip()) for part in value.split(',')]
    else:
        numbers = [parse_blue_ball(part) for part in value]
    if not numbers:
        raise ValueError('蓝球号码不能为空')
    if len(set(numbers)) != len(numbers):
        raise ValueError('蓝球号码不能重复')
    return numbers


# --- from ssq_paths.py ---
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')


# --- from ssq_prediction_workflow.py ---
@dataclass(frozen=True)
class ModelTrainingDependencies:
    train_models: Callable[..., Any]
    validate_models: Callable[..., Any]


# --- from ssq_prediction_workflow.py ---
@dataclass(frozen=True)
class CurrentSelectionDependencies:
    score_balls: Callable[..., Any]
    derive_rejection_seed: Callable[..., int]
    build_rejection_set: Callable[..., Any]
    get_omission: Callable[..., dict]
    generate_candidates: Callable[..., Any]
    collect_pipeline_stats: Callable[..., list]


# --- from ssq_prediction_workflow.py ---
def default_training_dependencies():
    return ModelTrainingDependencies(train_models=train_prediction_models, validate_models=validate_model_sets)


# --- from entry.py ---
def _impl_filter_pipeline_stats(combos, context, rule_definitions, rejection_set=None):
    """Measure incremental removal by each hard rule and anti-crowding."""
    checks = [(rule.name, lambda combo, current=rule: current.evaluator(combo, context)) for rule in rule_definitions if rule.hard]
    checks.append(('anti_crowding', lambda combo: rejection_set is None or combo not in rejection_set))
    remaining = list(combos)
    stats = []
    for name, check in checks:
        before = len(remaining)
        remaining = [combo for combo in remaining if check(combo)]
        stats.append({'rule': name, 'before': before, 'removed': before - len(remaining), 'remaining': len(remaining)})
    return stats


# --- from entry.py ---
def filter_pipeline_stats(combos, context, rejection_set=None):
    return _impl_filter_pipeline_stats(combos, context, RED_RULES, rejection_set)


# --- from ssq_prediction_workflow.py ---
def default_selection_dependencies():
    return CurrentSelectionDependencies(score_balls=run_strategy_and_get_scores, derive_rejection_seed=rejection_seed_for_issue, build_rejection_set=make_rejection_set, get_omission=get_omission, generate_candidates=generate_candidates, collect_pipeline_stats=filter_pipeline_stats)


# --- from ssq_prediction_workflow.py ---
def train_final_models(history, dependencies=None):
    """Train and validate the models used for the target issue."""
    if dependencies is None:
        dependencies = default_training_dependencies()
    models = dependencies.train_models(history.frame.iloc[5:].copy(), history.feature_columns, show_progress=True)
    try:
        dependencies.validate_models(*models)
    except ValueError as exc:
        raise SystemExit(f'错误: {exc}') from exc
    return models


# --- from ssq_workflow_models.py ---
@dataclass(frozen=True)
class CurrentSelection:
    red_scores: dict[int, float]
    recommended_blues: list[int]
    rejection_seed: int
    rule_context: RuleContext
    candidate_selection: Any
    pipeline_stats: list[dict]


# --- from ssq_prediction_workflow.py ---
def select_current_issue(history, options, params, models, dependencies=None):
    """Score the target issue and apply anti-crowding and hard rules."""
    if dependencies is None:
        dependencies = default_selection_dependencies()
    red_scores, blue_scores = dependencies.score_balls(history.frame, params, *models, history.feature_columns)
    recommended_blues = sorted(blue_scores, key=blue_scores.get, reverse=True)[:options.strategy_config.blue_count]
    print('\n[阶段 5/8] 正在从大底中生成组合并应用硬规则过滤...')
    config = options.strategy_config
    rejection_seed = dependencies.derive_rejection_seed(config.random_seed, history.target_issue)
    rejection_set = dependencies.build_rejection_set(config.rejection_lib_size, random.Random(rejection_seed))
    recent_draws = [set(draw) for draw in history.frame.iloc[-10:]['红球'].tolist()]
    context = RuleContext(omission_values=dependencies.get_omission(history.frame), recent_draws=recent_draws, last_draw=recent_draws[-1], previous_draw=recent_draws[-2])
    selection = dependencies.generate_candidates(CandidateGenerationRequest(red_scores=red_scores, context=context, rejection_set=rejection_set, config=config, mode=options.pool_mode, show_progress=True))
    print(f'已根据ML评分选出 {config.pool_size_red} 个红球大底: {list(selection.red_pool)}')
    print(f'过滤完成！共有 {len(selection.passed_combos)} 组号码通过硬规则检验。')
    return CurrentSelection(red_scores=red_scores, recommended_blues=recommended_blues, rejection_seed=rejection_seed, rule_context=context, candidate_selection=selection, pipeline_stats=dependencies.collect_pipeline_stats(selection.potential_combos, context, rejection_set))


# --- from ssq_rank_bands.py ---
RANK_BANDS = build_rank_bands()


# --- from ssq_recommendation_reporting.py ---
def format_recommendations_report(data):
    lines = ['\n--- 2. 推荐组合 ---']
    top_blue = data.recommended_blues[0] if data.recommended_blues else None
    single_combos = data.selection.recommendations if top_blue is not None else ()
    lines.append(f'\n【单式推荐 ({len(single_combos)}组)】')
    if single_combos:
        max_overlap = max((len(set(left) & set(right)) for left, right in combinations(single_combos, 2)), default=0)
        lines.append(f'  实际任意两注最大重合红球数: {max_overlap}')
        for index, combo in enumerate(single_combos, 1):
            lines.append(f'  组合 {index:>2}: 红球 {list(combo)!s:<24} 蓝球 [{top_blue:02d}]')
    else:
        lines.append('  - 未能生成足够的单式组合。')
    lines.append('\n【7+N 复式推荐 (1组)】')
    if data.best_7_reds and data.recommended_blues:
        lines.append(f'  红球: {list(data.best_7_reds[0][0])}')
        lines.append(f'  蓝球: {data.recommended_blues}')
    else:
        lines.append('  - 未能生成足够的复式组合。')
    return lines


# --- from ssq_rule_reporting.py ---
def format_rule_audit_report(data):
    lines = ['\n硬规则过滤统计 (按流水线累计):']
    for item in data.pipeline_stats:
        lines.append(f"  - {item['rule']:<22}: {item['before']} -> {item['remaining']} (remove {item['removed']})")
    inactive_rules = [item['rule'] for item in data.pipeline_stats if item['removed'] == 0]
    aggressive_rules = [item['rule'] for item in data.pipeline_stats if item['before'] and item['removed'] / item['before'] >= 0.5]
    if inactive_rules:
        lines.append(f"  提示：本轮未淘汰组合的规则: {', '.join(inactive_rules)}")
    if aggressive_rules:
        lines.append(f"  提示：淘汰比例达到或超过50%的规则: {', '.join(aggressive_rules)}")
    return lines


# --- from ssq_report_builder.py ---
def build_analysis_report(data):
    lines = ['=' * 60, '          双色球策略分析与推荐报告 (高级过滤版)', '=' * 60, '\n--- 0. 报告元数据 ---', f'Data_Basis_Issue: {data.latest_issue}', f'Prediction_Target_Issue: {data.target_issue}', f"报告生成时间: {data.generated_at.strftime('%Y-%m-%d %H:%M:%S')}"]
    if data.history_sha256:
        lines.append(f'Data_History_SHA256: {data.history_sha256}')
    for name, runtime_version in data.runtime_versions.items():
        lines.append(f'Runtime_{name}: {runtime_version}')
    if data.model_features:
        lines.append(f"Model_Features: {','.join(data.model_features)}")
    for name, value in data.model_training_params.items():
        lines.append(f'Model_LightGBM_{name}: {value}')
    lines.extend(format_rule_audit_report(data))
    lines.extend(format_recommendations_report(data))
    lines.append('\n' + '=' * 60 + '\n报告结束。祝您好运！\n' + '=' * 60)
    return '\n'.join(lines)


# --- from ssq_report_output.py ---
RUNTIME_PACKAGES = ('numpy', 'pandas', 'lightgbm', 'scikit-learn')


# --- from ssq_report_output.py ---
@dataclass(frozen=True)
class ReportOutputDependencies:
    build_report: Callable[..., str]
    write_text: Callable[..., Any]


# --- from ssq_report_output.py ---
def default_output_dependencies():
    return ReportOutputDependencies(build_report=build_analysis_report, write_text=atomic_write_text)


# --- from ssq_report_output.py ---
def collect_runtime_versions(packages: Sequence[str]=RUNTIME_PACKAGES):
    return {'python': platform.python_version(), **{package: version(package) for package in packages}}


# --- from ssq_report_output.py ---
def save_analysis_report(data, report_dir=REPORT_DIR, dependencies=None):
    """Build, print, and atomically persist an analysis report."""
    report_dir = report_dir if report_dir is not None else REPORT_DIR
    if dependencies is None:
        dependencies = default_output_dependencies()
    report = dependencies.build_report(data)
    print('\n\n' + report)
    try:
        os.makedirs(report_dir, exist_ok=True)
        timestamp = data.generated_at.strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(report_dir, f'ssq_analysis_output_{timestamp}.txt')
        dependencies.write_text(filepath, report)
    except OSError as exc:
        raise SystemExit(f'\n\n写入报告文件失败: {exc}') from exc
    print(f'\n\n报告已成功保存到文件: {filepath}')
    return filepath


# --- from ssq_rule_definitions.py ---
SOFT_FILTER_NAMES = tuple(rule.name for rule in RED_RULES if not rule.hard)


# --- from ssq_rule_validation.py ---
def validate_rule_definition(rule):
    if not isinstance(rule.hard, bool):
        raise TypeError(f'规则 {rule.name} 的 hard 标记必须为布尔值')
    if not callable(rule.evaluator):
        raise TypeError(f'规则 {rule.name} 的 evaluator 必须可调用')
    weight = rule.score_weight
    if isinstance(weight, bool) or not isinstance(weight, Real) or (not isfinite(weight)) or (weight < 0):
        raise ValueError(f'规则 {rule.name} 的评分权重必须为有限非负数')
    if rule.scorer is None and weight != 0:
        raise ValueError(f'规则 {rule.name} 有评分权重但缺少 scorer')
    if rule.scorer is not None and (not callable(rule.scorer) or weight == 0):
        raise ValueError(f'规则 {rule.name} 的 scorer 与评分权重不一致')
    if not rule.hard and rule.scorer is None:
        raise ValueError(f'软规则 {rule.name} 必须提供 scorer')


# --- from ssq_rule_validation.py ---
def validate_rule_names(rule_definitions):
    names = [rule.name for rule in rule_definitions]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError('规则名称必须为非空字符串')
    if len(names) != len(set(names)):
        raise ValueError('规则名称不能重复')


# --- from ssq_rule_validation.py ---
def validate_signal_weight(signal_weight):
    if isinstance(signal_weight, bool) or not isinstance(signal_weight, Real) or (not isfinite(signal_weight)) or (not 0 <= signal_weight <= 1):
        raise ValueError('基础排名信号权重必须为 0 到 1 之间的有限数值')


# --- from ssq_rule_validation.py ---
def validate_rule_registry(rule_definitions, signal_weight=COMBINATION_SIGNAL_WEIGHT):
    """Validate rule identity, behavior, and the normalized score budget."""
    rule_definitions = tuple(rule_definitions)
    validate_signal_weight(signal_weight)
    validate_rule_names(rule_definitions)
    for rule in rule_definitions:
        validate_rule_definition(rule)
    score_weight_total = sum((rule.score_weight for rule in rule_definitions), start=signal_weight)
    if not isclose(score_weight_total, 1.0):
        raise ValueError(f'组合评分总权重必须为 1，实际为 {score_weight_total}')
    return tuple(rule_definitions)


# --- from ssq_rules.py ---
def select_recommendations(passed_combos, red_scores, last_draw=None, previous_draw=None, limit=NUM_RECOMMENDATIONS, max_shared=MAX_SHARED_RED_BALLS, context=None):
    """Compatibility wrapper for request-based portfolio selection."""
    return select_recommendation_portfolio(RecommendationRequest(passed_combos=passed_combos, red_scores=red_scores, context=context or RuleContext(last_draw=last_draw, previous_draw=previous_draw), limit=limit, max_shared=max_shared))


# --- from ssq_workflow.py ---
@dataclass(frozen=True)
class AnalysisWorkflowDependencies:
    prepare_history: Callable[..., Any]
    evaluate_history: Callable[..., Any]
    train_models: Callable[..., Any]
    select_current: Callable[..., Any]
    display_candidates: Callable[..., Any]
    rank_duplex: Callable[..., Any]
    build_report_data: Callable[..., Any]
    now: Callable[..., Any]
    collect_versions: Callable[..., Any]
    save_report: Callable[..., Any]


# --- from ssq_workflow.py ---
def default_analysis_dependencies():
    return AnalysisWorkflowDependencies(prepare_history=prepare_history, evaluate_history=evaluate_history, train_models=train_final_models, select_current=select_current_issue, display_candidates=display_passed_combinations, rank_duplex=rank_duplex_candidates, build_report_data=build_analysis_report_data, now=local_now, collect_versions=collect_runtime_versions, save_report=save_analysis_report)


# --- from ssq_workflow.py ---
def run_analysis(options, dependencies=None):
    """Execute one complete analysis run and return the saved report path."""
    dependencies = dependencies or default_analysis_dependencies()
    print('=' * 70)
    print('         双色球策略分析器 v7.0')
    print('=' * 70)
    print('\n[阶段 1/8] 正在加载和处理历史数据...')
    history = dependencies.prepare_history()
    print('数据加载与特征工程完成。')
    print('\n[阶段 2/8] 正在执行历史规则审计与滚动回测...')
    evaluation = dependencies.evaluate_history(history, options)
    print('\n[阶段 3/8] 正在使用全部历史数据，训练用于最终预测的模型...')
    models = dependencies.train_models(history)
    print('\n[阶段 4/8] 正在为下一期号码进行机器学习评分...')
    current = dependencies.select_current(history, options, evaluation.loaded_params.values, models)
    print('\n[阶段 6/8] 正在整理通过硬规则的候选组合...')
    dependencies.display_candidates(current.candidate_selection.passed_combos, options.non_interactive)
    print('\n[阶段 7/8] 正在从最终组合中，生成高重合度的7红球大底...')
    best_7_reds = dependencies.rank_duplex(DuplexSelectionRequest(passed_combos=current.candidate_selection.passed_combos, red_pool=current.candidate_selection.red_pool, red_scores=current.red_scores, context=current.rule_context))
    print('\n[阶段 8/8] 正在生成最终推荐报告...')
    generated_at = dependencies.now()
    report_data = dependencies.build_report_data(history, evaluation, current, options, best_7_reds, generated_at, dependencies.collect_versions(), MODEL_TRAINING_PARAMS)
    return dependencies.save_report(report_data)


# --- from ssq_workflow.py ---
def main(argv=None):
    """Parse command-line arguments and run the analyzer workflow."""
    return run_analysis(parse_cli_options(argv))


# --- entry point: ssq_analyzer.py ---
if __name__ == '__main__':
    main()
