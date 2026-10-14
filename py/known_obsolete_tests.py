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

Entries are `<TestCaseClass>.<test_method>`, matched against the last two
components of a test id, so they survive the tests being regrouped into
per-pipeline files. Delete an entry if that behaviour is reintroduced.
"""

EXPECTED_OBSOLETE = {
    # re-export identity across the old module boundaries
    "DomainTests.test_core_preserves_domain_constant_exports",
    "ParsingTests.test_core_preserves_parsing_compatibility_exports",
    "AtomicFileIoTests.test_core_preserves_atomic_write_compatibility_export",
    "DrawScheduleTests.test_core_exports_schedule_functions",
    "ProjectPathTests.test_workflows_share_canonical_path_objects",
    # source-scanning guards on the old module layout
    "HistoryStoreTests.test_production_modules_do_not_depend_on_compatibility_facade",
    "CandidateGenerationTests.test_production_workflows_do_not_import_candidates_facade",
    "BallScoringTests.test_production_workflows_do_not_import_scoring_facade",
    "ModelTrainingTests.test_production_workflows_do_not_import_training_facade",
    # old facade delegation / patch-point wiring
    "BacktestEvaluationTests.test_backtesting_binds_patchable_candidate_generation",
    "BacktestRunnerTests.test_compatibility_wrapper_binds_legacy_patch_points",
    "RankingTests.test_rules_compatibility_wrapper_uses_its_active_registry",
    "LatestDrawTests.test_workflow_wrapper_delegates_its_validation_dependencies",
    "HistoryPreparationTests.test_workflow_loader_delegates_explicit_path",
    "HistoryEvaluationTests.test_pipeline_uses_injected_boundaries",
    "ScoringModuleTests.test_signal_fusion_preserves_declared_red_and_blue_weights",
    # the callable DI signature (request, dependencies) now lives on a private
    # implementation; the public name is the wrapper binding default deps
    "CandidateGenerationTests.test_pipeline_uses_injected_boundaries",
    # the old ssq_modeling facade re-exported its `lightgbm` import; after the
    # merge that name is no longer a module attribute, so patching it is moot
    "AnalyzerTests.test_ball_models_enable_lightgbm_deterministic_mode",
    "TrainingModuleTests.test_modeling_preserves_training_compatibility_exports",
    # the rule predicates keep a compatibility signature whose third positional
    # parameter differs from the concrete implementation
    "RuleExecutionTests.test_hard_filtering_ignores_soft_rule_for_admission",
    "RuleExecutionTests.test_explanation_includes_soft_rules_and_anti_crowding",
    "RuleExecutionTests.test_pipeline_statistics_are_incremental_and_hard_only",
    "RuleDiagnosticTests.test_diagnostics_evaluate_soft_rules_without_filtering_them",
}
