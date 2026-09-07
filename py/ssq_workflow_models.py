"""Data contracts shared by analysis workflow stages."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from ssq_config import LoadedStrategyParams
from ssq_rule_models import RuleContext


@dataclass(frozen=True)
class PreparedHistory:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    latest_issue: str
    target_issue: int
    sha256: str


@dataclass(frozen=True)
class HistoricalEvaluation:
    loaded_params: LoadedStrategyParams
    rule_coverage: dict
    hard_pipeline_coverage: dict
    backtests: dict[str, Any]
    selected_backtest: Any


@dataclass(frozen=True)
class CurrentSelection:
    red_scores: dict[int, float]
    recommended_blues: list[int]
    rejection_seed: int
    rule_context: RuleContext
    candidate_selection: Any
    pipeline_stats: list[dict]
