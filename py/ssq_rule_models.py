"""Data contracts shared by rule filtering and combination ranking."""

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from ssq_config import MAX_SHARED_RED_BALLS, NUM_RECOMMENDATIONS


@dataclass(frozen=True)
class RuleContext:
    omission_values: Mapping[int, int] = field(default_factory=dict)
    recent_draws: Sequence[Collection[int]] = field(default_factory=tuple)
    last_draw: Collection[int] | None = None
    previous_draw: Collection[int] | None = None


@dataclass(frozen=True)
class RuleDefinition:
    name: str
    hard: bool
    evaluator: Callable[[tuple[int, ...], RuleContext], bool]
    score_weight: float = 0.0
    scorer: Callable[[tuple[int, ...], RuleContext], float] | None = None


@dataclass(frozen=True)
class CombinationScoreContext:
    red_scores: Mapping[int, float]
    rank_center_scores: Mapping[int, float]
    rule_context: RuleContext


@dataclass(frozen=True)
class RecommendationRequest:
    passed_combos: Iterable[tuple[int, ...]]
    red_scores: Mapping[int, float]
    context: RuleContext
    limit: int = NUM_RECOMMENDATIONS
    max_shared: int = MAX_SHARED_RED_BALLS
