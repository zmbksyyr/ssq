"""Test compatibility layer for the merged three-file layout.

The historical test suite in `py/` was written against the previous 85-module
split and imports those module names directly. Some of those tests also assert
on the old *internal structure* (module boundaries, re-export identity,
`__all__` contents, "module X must not import facade Y").

This maps each retired module name onto the merged module that now owns its
behaviour, so the same test bodies exercise the merged code:

    py/ssq_data_prep.py    <- the data-acquisition pipeline   (was 15 modules)
    py/ssq_analysis.py     <- the analysis/backtest pipeline  (was 65 modules)
    py/ssq_bonus_check.py  <- the prize-check pipeline        (was 13 modules)

Tests that assert on the old internal structure are expected to fail: those
boundaries are exactly what the merge removed. Tests asserting behaviour must
keep passing. Use `python py/run_tests.py` to run the suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

PY = Path(__file__).resolve().parent
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

import ssq_analysis
import ssq_bonus_check
import ssq_data_prep

DATA_PREP = [
    "ssq_csv_store",
    "ssq_data_processor",
    "ssq_data_sources",
    "ssq_data_store",
    "ssq_data_workflow",
    "ssq_domain",
    "ssq_draw_data",
    "ssq_draw_schedule",
    "ssq_file_io",
    "ssq_history_store",
    "ssq_http",
    "ssq_parsing",
    "ssq_paths",
    "ssq_snapshot_validation",
    "ssq_source_comparison",
    "ssq_source_parsing",
]

ANALYSIS = [
    "ssq_analysis_result",
    "ssq_analyzer",
    "ssq_anti_crowding",
    "ssq_backtest_evaluation",
    "ssq_backtest_metrics",
    "ssq_backtest_models",
    "ssq_backtest_preparation",
    "ssq_backtest_reporting",
    "ssq_backtest_runner",
    "ssq_backtest_validation",
    "ssq_backtesting",
    "ssq_ball_scoring",
    "ssq_candidate_generation",
    "ssq_candidate_validation",
    "ssq_candidates",
    "ssq_cli",
    "ssq_combination_scoring",
    "ssq_config",
    "ssq_config_models",
    "ssq_console",
    "ssq_core",
    "ssq_duplex",
    "ssq_features",
    "ssq_history_evaluation",
    "ssq_history_preparation",
    "ssq_model_contracts",
    "ssq_model_training",
    "ssq_model_validation",
    "ssq_modeling",
    "ssq_prediction_workflow",
    "ssq_rank_bands",
    "ssq_ranking",
    "ssq_recommendation_portfolio",
    "ssq_recommendation_reporting",
    "ssq_recommendation_validation",
    "ssq_red_pool",
    "ssq_report_builder",
    "ssq_report_models",
    "ssq_report_output",
    "ssq_reporting",
    "ssq_rule_auditing",
    "ssq_rule_definitions",
    "ssq_rule_diagnostics",
    "ssq_rule_execution",
    "ssq_rule_functions",
    "ssq_rule_models",
    "ssq_rule_registry",
    "ssq_rule_reporting",
    "ssq_rule_scoring",
    "ssq_rule_validation",
    "ssq_rules",
    "ssq_score_adjustments",
    "ssq_score_signals",
    "ssq_scoring",
    "ssq_selection",
    "ssq_selection_models",
    "ssq_strategy_params",
    "ssq_strategy_validation",
    "ssq_training",
    "ssq_workflow",
    "ssq_workflow_models",
]

BONUS_CHECK = [
    "ssq_bet_parsing",
    "ssq_bonus_calculation",
    "ssq_bonus_reporting",
    "ssq_bonus_workflow",
    "ssq_latest_draw",
    "ssq_prize_calculation",
    "ssq_prizes",
    "ssq_report_discovery",
]

for _name in DATA_PREP:
    sys.modules.setdefault(_name, ssq_data_prep)
for _name in ANALYSIS:
    sys.modules.setdefault(_name, ssq_analysis)
for _name in BONUS_CHECK:
    sys.modules.setdefault(_name, ssq_bonus_check)
