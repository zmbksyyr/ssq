"""Data contract for a complete analysis report."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class AnalysisReportData:
    latest_issue: str
    target_issue: int
    generated_at: datetime
    params_loaded: bool
    params: dict
    config: Any
    rejection_seed: int
    backtest: Any
    backtests: dict[str, Any]
    pool_mode: str
    rank_band_widths: dict[str, int]
    rank_band_labels: dict[str, str]
    pipeline_stats: list[dict]
    rule_coverage: dict
    hard_pipeline_coverage: dict
    rule_audit_periods: int
    selection: Any
    recommended_blues: list[int]
    best_7_reds: list
    runtime_versions: dict[str, str] = field(default_factory=dict)
    history_sha256: str = ''
    model_features: tuple[str, ...] = ()
    model_training_params: Mapping[str, Any] = field(default_factory=dict)
