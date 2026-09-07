"""Final model training and current-issue candidate selection stages."""

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ssq_anti_crowding import make_rejection_set, rejection_seed_for_issue
from ssq_candidates import generate_candidates
from ssq_rule_models import RuleContext
from ssq_rule_registry import filter_pipeline_stats
from ssq_scoring import get_omission, run_strategy_and_get_scores
from ssq_selection_models import CandidateGenerationRequest
from ssq_training import train_prediction_models, validate_model_sets
from ssq_workflow_models import CurrentSelection


@dataclass(frozen=True)
class ModelTrainingDependencies:
    train_models: Callable[..., Any]
    validate_models: Callable[..., Any]


@dataclass(frozen=True)
class CurrentSelectionDependencies:
    score_balls: Callable[..., Any]
    derive_rejection_seed: Callable[..., int]
    build_rejection_set: Callable[..., Any]
    get_omission: Callable[..., dict]
    generate_candidates: Callable[..., Any]
    collect_pipeline_stats: Callable[..., list]


def default_training_dependencies():
    return ModelTrainingDependencies(
        train_models=train_prediction_models,
        validate_models=validate_model_sets,
    )


def default_selection_dependencies():
    return CurrentSelectionDependencies(
        score_balls=run_strategy_and_get_scores,
        derive_rejection_seed=rejection_seed_for_issue,
        build_rejection_set=make_rejection_set,
        get_omission=get_omission,
        generate_candidates=generate_candidates,
        collect_pipeline_stats=filter_pipeline_stats,
    )


def train_final_models(history, dependencies=None):
    """Train and validate the models used for the target issue."""
    if dependencies is None:
        dependencies = default_training_dependencies()
    models = dependencies.train_models(
        history.frame.iloc[5:].copy(),
        history.feature_columns,
        show_progress=True,
    )
    try:
        dependencies.validate_models(*models)
    except ValueError as exc:
        raise SystemExit(f'错误: {exc}') from exc
    return models


def select_current_issue(history, options, params, models, dependencies=None):
    """Score the target issue and apply anti-crowding and hard rules."""
    if dependencies is None:
        dependencies = default_selection_dependencies()
    red_scores, blue_scores = dependencies.score_balls(
        history.frame,
        params,
        *models,
        history.feature_columns,
    )
    recommended_blues = sorted(
        blue_scores,
        key=blue_scores.get,
        reverse=True,
    )[:options.strategy_config.blue_count]

    print('\n[阶段 5/8] 正在从大底中生成组合并应用硬规则过滤...')
    config = options.strategy_config
    rejection_seed = dependencies.derive_rejection_seed(
        config.random_seed,
        history.target_issue,
    )
    rejection_set = dependencies.build_rejection_set(
        config.rejection_lib_size,
        random.Random(rejection_seed),
    )
    recent_draws = [set(draw) for draw in history.frame.iloc[-10:]['红球'].tolist()]
    context = RuleContext(
        omission_values=dependencies.get_omission(history.frame),
        recent_draws=recent_draws,
        last_draw=recent_draws[-1],
        previous_draw=recent_draws[-2],
    )
    selection = dependencies.generate_candidates(CandidateGenerationRequest(
        red_scores=red_scores,
        context=context,
        rejection_set=rejection_set,
        config=config,
        mode=options.pool_mode,
        show_progress=True,
    ))
    print(
        f'已根据ML评分选出 {config.pool_size_red} 个红球大底: '
        f'{list(selection.red_pool)}'
    )
    print(f'过滤完成！共有 {len(selection.passed_combos)} 组号码通过硬规则检验。')
    return CurrentSelection(
        red_scores=red_scores,
        recommended_blues=recommended_blues,
        rejection_seed=rejection_seed,
        rule_context=context,
        candidate_selection=selection,
        pipeline_stats=dependencies.collect_pipeline_stats(
            selection.potential_combos,
            context,
            rejection_set,
        ),
    )
