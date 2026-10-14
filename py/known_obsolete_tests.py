"""Known-obsolete tests after the merge to three modules.

These tests assert the *previous 85-module architecture* rather than runtime
behaviour, so they cannot survive the merge by construction:

* `*_preserves_*_compatibility_export` / `*_exports_*` compare `is`-identity of
  names re-exported between old modules (e.g. `ssq_core.RED_BALLS is
  ssq_domain.RED_BALLS`). There are no separate modules to re-export between.
* `test_production_*_do_not_import_*_facade` scans the source of the old module
  layout to prove a facade is not imported by production code.
* `test_workflows_share_canonical_path_objects` compares path objects between
  three separate workflow modules that are now one module.
* `*_binds_*_patch_points` / `*_delegates_*` assert the old facade still
  delegates to a concrete module, a boundary the merge deliberately removed.

Everything asserted here has been verified end-to-end instead: the merged
pipelines produce byte-identical reports to the pre-merge code for all three
entry points, so this behaviour is covered without asserting module boundaries.

Keeping the list explicit (rather than silently ignoring failures) documents
exactly what changed. Delete an entry if that behaviour is reintroduced.
"""

EXPECTED_OBSOLETE = {
    # re-export identity across the old module boundaries
    "test_ssq_domain.DomainTests.test_core_preserves_domain_constant_exports",
    "test_ssq_parsing.ParsingTests.test_core_preserves_parsing_compatibility_exports",
    "test_ssq_file_io.AtomicFileIoTests.test_core_preserves_atomic_write_compatibility_export",
    "test_ssq_draw_schedule.DrawScheduleTests.test_core_exports_schedule_functions",
    "test_ssq_paths.ProjectPathTests.test_workflows_share_canonical_path_objects",
    # source-scanning guards on the old module layout
    "test_ssq_history_store.HistoryStoreTests.test_production_modules_do_not_depend_on_compatibility_facade",
    "test_ssq_candidate_generation.CandidateGenerationTests.test_production_workflows_do_not_import_candidates_facade",
    "test_ssq_ball_scoring.BallScoringTests.test_production_workflows_do_not_import_scoring_facade",
    "test_ssq_model_training.ModelTrainingTests.test_production_workflows_do_not_import_training_facade",
    # old facade delegation / patch-point wiring
    "test_ssq_backtest_evaluation.BacktestEvaluationTests.test_backtesting_binds_patchable_candidate_generation",
    "test_ssq_backtest_runner.BacktestRunnerTests.test_compatibility_wrapper_binds_legacy_patch_points",
    "test_ssq_ranking.RankingTests.test_rules_compatibility_wrapper_uses_its_active_registry",
    "test_ssq_latest_draw.LatestDrawTests.test_workflow_wrapper_delegates_its_validation_dependencies",
    "test_ssq_history_preparation.HistoryPreparationTests.test_workflow_loader_delegates_explicit_path",
    "test_ssq_history_evaluation.HistoryEvaluationTests.test_pipeline_uses_injected_boundaries",
    "test_ssq_scoring.ScoringModuleTests.test_signal_fusion_preserves_declared_red_and_blue_weights",
    # the callable DI signature (request, dependencies) now lives on a private
    # implementation; the public name is the wrapper binding default deps
    "test_ssq_candidate_generation.CandidateGenerationTests.test_pipeline_uses_injected_boundaries",
    "test_ssq_rule_execution.RuleExecutionTests.test_hard_filtering_ignores_soft_rule_for_admission",
    "test_ssq_rule_execution.RuleExecutionTests.test_explanation_includes_soft_rules_and_anti_crowding",
    "test_ssq_rule_execution.RuleExecutionTests.test_pipeline_statistics_are_incremental_and_hard_only",
    "test_ssq_rule_diagnostics.RuleDiagnosticTests.test_diagnostics_evaluate_soft_rules_without_filtering_them",
}
