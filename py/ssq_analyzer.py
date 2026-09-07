"""Command-line entry point and compatibility facade for the SSQ analyzer."""

from ssq_anti_crowding import (  # noqa: F401 - compatibility exports
    make_rejection_set,
    rejection_seed_for_issue,
)
from ssq_backtesting import (  # noqa: F401 - compatibility exports
    FILTER_NAMES,
    BacktestAccumulator,
    BacktestIssue,
    BacktestRequest,
    BacktestResult,
    BacktestSelectionInputs,
    audit_historical_hard_pipeline,
    audit_historical_rule_coverage,
    evaluate_backtest_mode,
    historical_rule_context,
    record_backtest_selection,
    run_backtest,
    run_full_backtest,
    validate_backtest_request,
)
from ssq_candidates import (  # noqa: F401 - compatibility exports
    build_red_pool,
    count_actual_reds_by_rank_band,
    generate_candidates,
    generate_red_candidates,
    validate_candidate_generation_request,
)
from ssq_config import (  # noqa: F401 - compatibility exports
    BACKTEST_PERIODS,
    COUNTDOWN_SECONDS,
    DEFAULT_PARAMS,
    DEFAULT_STRATEGY_CONFIG,
    INTERACTIVE_THRESHOLD,
    MAX_SHARED_RED_BALLS,
    NUM_BLUE_BALLS,
    NUM_RECOMMENDATIONS,
    POOL_SIZE_RED,
    RANDOM_SEED,
    RED_HIGH_COUNT,
    RED_LOW_COUNT,
    RED_POOL_MODES,
    REJECTION_LIB_SIZE,
    REJECTION_SEED_MULTIPLIER,
    RULE_AUDIT_PERIODS,
    TOTAL_RED_COMBINATIONS,
    AnalyzerOptions,
    LoadedStrategyParams,
    StrategyConfig,
    build_argument_parser,
    parse_cli_options,
    validate_strategy_params,
)
from ssq_duplex import (  # noqa: F401 - compatibility exports
    find_best_7_red_combinations,
    rank_duplex_candidates,
    validate_duplex_selection_request,
)
from ssq_features import (  # noqa: F401 - compatibility exports
    FEATURE_COLUMNS,
    feature_engineer,
    validate_feature_columns,
)
from ssq_rank_bands import (  # noqa: F401 - compatibility exports
    RANK_BAND_WIDTHS,
    RANK_BANDS,
    build_rank_band_labels,
    build_rank_band_widths,
    build_rank_bands,
)
from ssq_reporting import (  # noqa: F401 - compatibility exports
    AnalysisReportData,
    build_analysis_report,
)
from ssq_rule_models import (  # noqa: F401 - compatibility exports
    CombinationScoreContext,
    RecommendationRequest,
    RuleContext,
    RuleDefinition,
)
from ssq_rules import (  # noqa: F401 - compatibility exports
    filter_pipeline_stats,
    select_recommendation_portfolio,
    select_recommendations,
    validate_recommendation_request,
)
from ssq_scoring import (  # noqa: F401 - compatibility exports
    apply_red_score_adjustments,
    get_omission,
    get_weighted_frequency,
    run_strategy_and_get_scores,
)
from ssq_selection import (  # noqa: F401 - compatibility exports
    passes_red_filters,
)
from ssq_selection_models import (  # noqa: F401 - compatibility exports
    CandidateGenerationRequest,
    DuplexSelectionRequest,
    RedCandidateSelection,
)
from ssq_training import (  # noqa: F401 - compatibility exports
    MODEL_TRAINING_PARAMS,
    BallModelSpec,
    predict_positive_probability,
    train_ball_models,
    train_models_for_spec,
    train_prediction_models,
    validate_model_sets,
)
from ssq_workflow import (  # noqa: F401 - compatibility exports
    CSV_PATH,
    PARAMS_JSON_PATH,
    PROJECT_ROOT,
    REPORT_DIR,
    collect_runtime_versions,
    get_user_input_with_timeout,
    is_confirmation_input,
    load_and_preprocess_data,
    load_strategy_params,
    main,
    run_analysis,
)

if __name__ == '__main__':
    main()
