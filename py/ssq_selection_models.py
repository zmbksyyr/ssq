"""Request and result contracts for red-ball selection."""

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

from ssq_config import DEFAULT_STRATEGY_CONFIG, StrategyConfig
from ssq_rule_models import RuleContext


@dataclass(frozen=True)
class RedCandidateSelection:
    red_pool: tuple[int, ...]
    potential_combos: tuple[tuple[int, ...], ...]
    passed_combos: tuple[tuple[int, ...], ...]
    recommendations: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class CandidateGenerationRequest:
    red_scores: Mapping[int, float]
    context: RuleContext
    rejection_set: Collection[tuple[int, ...]] | None
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG
    mode: str = 'mixed'
    show_progress: bool = False


@dataclass(frozen=True)
class DuplexSelectionRequest:
    passed_combos: Collection[tuple[int, ...]]
    red_pool: Collection[int]
    red_scores: Mapping[int, float] | None = None
    context: RuleContext = field(default_factory=RuleContext)
