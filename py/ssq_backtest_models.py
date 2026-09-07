"""Data contracts for rolling strategy backtests."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ssq_config import DEFAULT_STRATEGY_CONFIG, StrategyConfig
from ssq_rule_models import RuleContext


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


@dataclass(frozen=True)
class BacktestRunContext:
    params: dict
    feature_columns: Sequence[str]
    config: StrategyConfig


@dataclass(frozen=True)
class BacktestRequest:
    params: dict
    feature_columns: Sequence[str]
    num_periods: int
    pool_modes: Sequence[str] = ('mixed',)
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG


@dataclass(frozen=True)
class PreparedBacktestIssue:
    issue: BacktestIssue
    selection_inputs: BacktestSelectionInputs
