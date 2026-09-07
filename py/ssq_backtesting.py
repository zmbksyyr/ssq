"""Historical rule audits and leakage-free rolling strategy backtests."""

import random
from collections import Counter
from dataclasses import dataclass, field

from ssq_config import (
    DEFAULT_STRATEGY_CONFIG,
    RED_POOL_MODES,
    RULE_AUDIT_PERIODS,
    StrategyConfig,
    normalize_integer_param,
    validate_strategy_params,
)
from ssq_core import PRIZE_RULES
from ssq_modeling import (
    get_omission,
    run_strategy_and_get_scores,
    train_prediction_models,
    validate_model_sets,
)
from ssq_rules import FILTER_NAMES, RED_RULES, RuleContext, explain_filter_failures
from ssq_selection import (
    RANK_BAND_WIDTHS,
    build_rank_band_widths,
    count_actual_reds_by_rank_band,
    generate_red_candidates,
    make_rejection_set,
    rejection_seed_for_issue,
)
from tqdm import tqdm


@dataclass(frozen=True)
class BacktestResult:
    periods: int
    active_periods: int
    tickets: int
    cost: int
    winnings: int
    prize_counts: Counter
    evaluated_periods: int = 0
    pool_red_hits: int = 0
    ticket_red_hit_counts: Counter = field(default_factory=Counter)
    candidate_tickets: int = 0
    candidate_red_hit_counts: Counter = field(default_factory=Counter)
    blue_hit_periods: int = 0
    rank_band_hits: Counter = field(default_factory=Counter)
    rank_band_widths: dict[str, int] = field(
        default_factory=lambda: RANK_BAND_WIDTHS.copy()
    )
    windows: dict[str, 'BacktestResult'] = field(default_factory=dict)

    @property
    def profit(self):
        return self.winnings - self.cost

    @property
    def roi(self):
        return self.winnings / self.cost if self.cost else 0.0

    @property
    def average_pool_red_hits(self):
        if not self.evaluated_periods:
            return 0.0
        return self.pool_red_hits / self.evaluated_periods

    @property
    def average_ticket_red_hits(self):
        if not self.tickets:
            return 0.0
        total_hits = sum(
            hits * count for hits, count in self.ticket_red_hit_counts.items()
        )
        return total_hits / self.tickets

    @property
    def three_plus_red_tickets(self):
        return sum(
            count for hits, count in self.ticket_red_hit_counts.items()
            if hits >= 3
        )

    @property
    def three_plus_red_rate(self):
        return self.three_plus_red_tickets / self.tickets if self.tickets else 0.0

    @property
    def average_candidate_red_hits(self):
        if not self.candidate_tickets:
            return 0.0
        total_hits = sum(
            hits * count for hits, count in self.candidate_red_hit_counts.items()
        )
        return total_hits / self.candidate_tickets

    @property
    def candidate_three_plus_red_rate(self):
        if not self.candidate_tickets:
            return 0.0
        three_plus = sum(
            count for hits, count in self.candidate_red_hit_counts.items()
            if hits >= 3
        )
        return three_plus / self.candidate_tickets

    @property
    def ranking_red_hit_delta(self):
        return self.average_ticket_red_hits - self.average_candidate_red_hits

    @property
    def ranking_three_plus_delta(self):
        return self.three_plus_red_rate - self.candidate_three_plus_red_rate

    @property
    def blue_hit_rate(self):
        if not self.evaluated_periods:
            return 0.0
        return self.blue_hit_periods / self.evaluated_periods

    def rank_band_rate(self, band):
        total_actual_reds = self.evaluated_periods * 6
        if not total_actual_reds:
            return 0.0
        return self.rank_band_hits[band] / total_actual_reds

    def rank_band_lift(self, band):
        expected_rate = (
            self.rank_band_widths[band] / sum(self.rank_band_widths.values())
        )
        return self.rank_band_rate(band) / expected_rate if expected_rate else 0.0


@dataclass
class BacktestAccumulator:
    prize_counts: Counter = field(default_factory=Counter)
    cost: int = 0
    winnings: int = 0
    active_periods: int = 0
    evaluated_periods: int = 0
    tickets: int = 0
    pool_red_hits: int = 0
    ticket_red_hit_counts: Counter = field(default_factory=Counter)
    candidate_tickets: int = 0
    candidate_red_hit_counts: Counter = field(default_factory=Counter)
    blue_hit_periods: int = 0
    rank_band_hits: Counter = field(default_factory=Counter)
    rank_band_widths: dict[str, int] = field(
        default_factory=lambda: RANK_BAND_WIDTHS.copy()
    )

    def to_result(self, periods, windows=None):
        return BacktestResult(
            periods=periods,
            active_periods=self.active_periods,
            tickets=self.tickets,
            cost=self.cost,
            winnings=self.winnings,
            prize_counts=Counter(self.prize_counts),
            evaluated_periods=self.evaluated_periods,
            pool_red_hits=self.pool_red_hits,
            ticket_red_hit_counts=Counter(self.ticket_red_hit_counts),
            candidate_tickets=self.candidate_tickets,
            candidate_red_hit_counts=Counter(self.candidate_red_hit_counts),
            blue_hit_periods=self.blue_hit_periods,
            rank_band_hits=Counter(self.rank_band_hits),
            rank_band_widths=self.rank_band_widths.copy(),
            windows=dict(windows or {}),
        )


@dataclass(frozen=True)
class BacktestIssue:
    actual_reds: frozenset[int]
    actual_blue: int
    recommended_blue: int
    rank_band_hits: Counter


@dataclass(frozen=True)
class BacktestSelectionInputs:
    red_scores: dict[int, float]
    context: RuleContext
    rejection_set: set[tuple[int, ...]]
    config: StrategyConfig


def historical_rule_context(full_df, index):
    """Build rule inputs using only draws before the audited issue."""
    history = full_df.iloc[:index]
    recent = [set(draw) for draw in history.iloc[-10:]['红球']]
    return RuleContext(
        omission_values=get_omission(history),
        recent_draws=recent,
        last_draw=recent[-1],
        previous_draw=recent[-2],
    )


def audit_historical_rule_coverage(full_df, periods=RULE_AUDIT_PERIODS):
    """Measure how often each strategy rule accepts actual historical draws."""
    if periods <= 0 or len(full_df) < 11:
        return {
            name: {'passed': 0, 'total': 0, 'rate': 0.0}
            for name in FILTER_NAMES
        }
    start = max(10, len(full_df) - periods)
    passed_counts = Counter()
    total = 0
    for index in range(start, len(full_df)):
        combo = tuple(full_df.iloc[index]['红球'])
        failures = set(explain_filter_failures(
            combo,
            historical_rule_context(full_df, index),
        ))
        for name in FILTER_NAMES:
            if name not in failures:
                passed_counts[name] += 1
        total += 1
    return {
        name: {
            'passed': passed_counts[name],
            'total': total,
            'rate': passed_counts[name] / total if total else 0.0,
        }
        for name in FILTER_NAMES
    }


def audit_historical_hard_pipeline(full_df, periods=RULE_AUDIT_PERIODS):
    """Measure cumulative survival of actual draws through ordered hard rules."""
    hard_rules = [rule for rule in RED_RULES if rule.hard]
    if periods <= 0 or len(full_df) < 11:
        return {
            'total': 0,
            'passed': 0,
            'rate': 0.0,
            'stages': [
                {'rule': rule.name, 'before': 0, 'removed': 0, 'remaining': 0}
                for rule in hard_rules
            ],
        }

    start = max(10, len(full_df) - periods)
    total = len(full_df) - start
    remaining_counts = Counter()
    for index in range(start, len(full_df)):
        combo = tuple(full_df.iloc[index]['红球'])
        context = historical_rule_context(full_df, index)
        for rule in hard_rules:
            if not rule.evaluator(combo, context):
                break
            remaining_counts[rule.name] += 1

    stages = []
    before = total
    for rule in hard_rules:
        remaining = remaining_counts[rule.name]
        stages.append({
            'rule': rule.name,
            'before': before,
            'removed': before - remaining,
            'remaining': remaining,
        })
        before = remaining
    return {
        'total': total,
        'passed': before,
        'rate': before / total if total else 0.0,
        'stages': stages,
    }


def evaluate_backtest_mode(
    mode,
    current,
    issue,
    selection_inputs,
    additional_accumulators=(),
):
    """Evaluate one pool mode for one historical issue."""
    selection = generate_red_candidates(
        selection_inputs.red_scores,
        selection_inputs.context,
        selection_inputs.rejection_set,
        config=selection_inputs.config,
        mode=mode,
    )
    red_hits_by_combo = None
    for accumulator in (current, *additional_accumulators):
        red_hits_by_combo = record_backtest_selection(
            accumulator,
            selection,
            issue,
            red_hits_by_combo,
        )
    return selection


def record_backtest_selection(
    current,
    selection,
    issue,
    red_hits_by_combo=None,
):
    """Accumulate one already-generated selection into a result window."""
    current.evaluated_periods += 1
    current.pool_red_hits += len(set(selection.red_pool) & issue.actual_reds)
    current.rank_band_hits.update(issue.rank_band_hits)
    if issue.recommended_blue == issue.actual_blue:
        current.blue_hit_periods += 1

    if not selection.passed_combos:
        return {}

    if red_hits_by_combo is None:
        red_hits_by_combo = {
            combo: len(set(combo) & issue.actual_reds)
            for combo in selection.passed_combos
        }
    current.candidate_tickets += len(selection.passed_combos)
    current.candidate_red_hit_counts.update(red_hits_by_combo.values())
    current.active_periods += 1
    current.tickets += len(selection.recommendations)
    current.cost += len(selection.recommendations) * 2
    blue_hits = int(issue.recommended_blue == issue.actual_blue)
    for combo in selection.recommendations:
        red_hits = red_hits_by_combo[combo]
        current.ticket_red_hit_counts[red_hits] += 1
        hit_key = (red_hits, blue_hits)
        prize = PRIZE_RULES.get(hit_key, 0)
        if prize > 0:
            current.winnings += prize
            current.prize_counts[hit_key] += 1
    return red_hits_by_combo


def validate_backtest_request(num_periods, pool_modes, config):
    """Normalize and validate public backtest controls before expensive work."""
    num_periods = normalize_integer_param('num_periods', num_periods)
    if num_periods < 0:
        raise ValueError('num_periods 不能为负数')
    if isinstance(pool_modes, str):
        raise TypeError('pool_modes 必须为候选池模式序列，不能是字符串')
    try:
        pool_modes = tuple(pool_modes)
    except TypeError as exc:
        raise TypeError('pool_modes 必须为候选池模式序列') from exc
    if not pool_modes:
        raise ValueError('pool_modes 不能为空')
    if len(pool_modes) != len(set(pool_modes)):
        raise ValueError('pool_modes 不能包含重复模式')
    invalid_modes = sorted(set(pool_modes) - set(RED_POOL_MODES))
    if invalid_modes:
        raise ValueError(f'未知候选池模式: {invalid_modes}')
    if not isinstance(config, StrategyConfig):
        raise TypeError('config 必须为 StrategyConfig')
    return num_periods, pool_modes


def run_full_backtest(
    full_df,
    params,
    feature_columns,
    num_periods,
    pool_modes=('mixed',),
    config=DEFAULT_STRATEGY_CONFIG,
):
    """Run a rolling backtest that retrains using only earlier draws."""
    num_periods, pool_modes = validate_backtest_request(
        num_periods,
        pool_modes,
        config,
    )
    params = validate_strategy_params(params)
    print('\n' + '=' * 70)
    print(f'        最近 {num_periods} 期完整策略滚动回测')
    print('=' * 70)

    minimum_history = num_periods + 50
    if len(full_df) < minimum_history:
        print(f'历史数据不足 {minimum_history} 期，无法执行回测。跳过此步骤。')
        return {
            mode: BacktestResult(
                0, 0, 0, 0, 0, Counter(),
                rank_band_widths=build_rank_band_widths(config),
            )
            for mode in pool_modes
        }

    rank_band_widths = build_rank_band_widths(config)
    metrics = {
        mode: BacktestAccumulator(rank_band_widths=rank_band_widths.copy())
        for mode in pool_modes
    }
    window_metrics = {
        name: {
            mode: BacktestAccumulator(rank_band_widths=rank_band_widths.copy())
            for mode in pool_modes
        }
        for name in ('earlier', 'recent')
    }
    backtest_range = range(len(full_df) - num_periods, len(full_df))
    split_offset = len(backtest_range) // 2
    window_periods = {
        'earlier': split_offset,
        'recent': len(backtest_range) - split_offset,
    }

    with tqdm(total=len(backtest_range), desc='执行严谨回测', ncols=80) as progress:
        for offset, index in enumerate(backtest_range):
            history = full_df.iloc[:index]
            actual_draw = full_df.iloc[index]
            actual_red_set = set(actual_draw['红球'])
            actual_blue = actual_draw['蓝球']

            training_data = history.iloc[5:].copy()
            if len(training_data) < 20:
                progress.update(1)
                continue

            red_models, blue_models = train_prediction_models(
                training_data,
                feature_columns,
            )
            try:
                validate_model_sets(red_models, blue_models)
            except ValueError:
                progress.update(1)
                continue

            red_scores, blue_scores = run_strategy_and_get_scores(
                history,
                params,
                red_models,
                blue_models,
                feature_columns,
            )
            recommended_blue = max(blue_scores, key=blue_scores.get)
            rank_band_hits = count_actual_reds_by_rank_band(
                red_scores,
                actual_red_set,
                config,
            )
            rejection_seed = rejection_seed_for_issue(
                config.random_seed,
                actual_draw['期号'],
            )
            rejection_set = make_rejection_set(
                config.rejection_lib_size,
                random.Random(rejection_seed),
            )
            context = historical_rule_context(full_df, index)
            window_name = 'earlier' if offset < split_offset else 'recent'
            issue = BacktestIssue(
                actual_reds=frozenset(actual_red_set),
                actual_blue=actual_blue,
                recommended_blue=recommended_blue,
                rank_band_hits=rank_band_hits,
            )
            selection_inputs = BacktestSelectionInputs(
                red_scores=red_scores,
                context=context,
                rejection_set=rejection_set,
                config=config,
            )

            for mode in pool_modes:
                evaluate_backtest_mode(
                    mode,
                    metrics[mode],
                    issue,
                    selection_inputs,
                    additional_accumulators=(
                        window_metrics[window_name][mode],
                    ),
                )
            progress.update(1)

    print('回测完成。\n')
    return {
        mode: value.to_result(len(backtest_range), windows=(
            {
                name: window_metrics[name][mode].to_result(window_periods[name])
                for name in ('earlier', 'recent')
            }
            if len(backtest_range) >= 2 else {}
        ))
        for mode, value in metrics.items()
    }
