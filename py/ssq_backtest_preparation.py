"""Leakage-free preparation of one historical issue for backtesting."""

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ssq_anti_crowding import make_rejection_set, rejection_seed_for_issue
from ssq_backtest_models import (
    BacktestIssue,
    BacktestSelectionInputs,
    PreparedBacktestIssue,
)
from ssq_ball_scoring import run_strategy_and_get_scores
from ssq_candidates import count_actual_reds_by_rank_band
from ssq_rule_auditing import historical_rule_context
from ssq_training import train_prediction_models, validate_model_sets


@dataclass(frozen=True)
class BacktestPreparationDependencies:
    train_models: Callable[..., Any]
    validate_models: Callable[..., Any]
    score_balls: Callable[..., Any]
    count_rank_bands: Callable[..., Any]
    build_rule_context: Callable[..., Any]
    derive_rejection_seed: Callable[..., Any]
    build_rejection_set: Callable[..., Any]


def default_preparation_dependencies():
    return BacktestPreparationDependencies(
        train_models=train_prediction_models,
        validate_models=validate_model_sets,
        score_balls=run_strategy_and_get_scores,
        count_rank_bands=count_actual_reds_by_rank_band,
        build_rule_context=historical_rule_context,
        derive_rejection_seed=rejection_seed_for_issue,
        build_rejection_set=make_rejection_set,
    )


def prepare_backtest_issue(full_df, index, run_context, dependencies=None):
    """Build model and selection inputs using only draws before one issue."""
    if dependencies is None:
        dependencies = default_preparation_dependencies()
    history = full_df.iloc[:index]
    training_data = history.iloc[5:].copy()
    if len(training_data) < 20:
        return None

    red_models, blue_models = dependencies.train_models(
        training_data,
        run_context.feature_columns,
    )
    try:
        dependencies.validate_models(red_models, blue_models)
    except ValueError:
        return None

    red_scores, blue_scores = dependencies.score_balls(
        history,
        run_context.params,
        red_models,
        blue_models,
        run_context.feature_columns,
    )
    actual_draw = full_df.iloc[index]
    actual_reds = frozenset(actual_draw['红球'])
    recommended_blue = max(blue_scores, key=blue_scores.get)
    rank_band_hits = dependencies.count_rank_bands(
        red_scores,
        actual_reds,
        run_context.config,
    )
    rejection_seed = dependencies.derive_rejection_seed(
        run_context.config.random_seed,
        actual_draw['期号'],
    )
    rejection_set = dependencies.build_rejection_set(
        run_context.config.rejection_lib_size,
        random.Random(rejection_seed),
    )
    return PreparedBacktestIssue(
        issue=BacktestIssue(
            actual_reds=actual_reds,
            actual_blue=actual_draw['蓝球'],
            recommended_blue=recommended_blue,
            rank_band_hits=rank_band_hits,
        ),
        selection_inputs=BacktestSelectionInputs(
            red_scores=red_scores,
            context=dependencies.build_rule_context(full_df, index),
            rejection_set=rejection_set,
            config=run_context.config,
        ),
    )
