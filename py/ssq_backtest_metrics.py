"""Backtest metric snapshots and mutable accumulation state."""

from collections import Counter
from dataclasses import dataclass, field

from ssq_rank_bands import RANK_BAND_WIDTHS


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
