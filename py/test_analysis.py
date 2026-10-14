"""Consolidated analysis tests (merged from 54 module test files).

Originally one test module per production module. The production modules are now
merged into py/ssq_analysis.py, so the tests are grouped the same way. Retired module
names still resolve through ssq_compat, which run_tests.py sets up before these
modules are imported.

Run with: python py/run_tests.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ast
import inspect
import io
import json
import random
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import UTC, date, datetime
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import lightgbm
import numpy
import pandas
import ssq_analyzer as analyzer
import ssq_anti_crowding as anti_crowding
import ssq_bonus_calculation as bonus_calculation
import ssq_candidate_generation as candidate_generation
import ssq_candidates as candidates
import ssq_cli
import ssq_config as config
import ssq_console as console
import ssq_duplex as duplex
import ssq_features as features
import ssq_history_preparation as history_preparation
import ssq_model_contracts as model_contracts
import ssq_model_training as model_training
import ssq_modeling as modeling
import ssq_prediction_workflow as prediction_workflow
import ssq_rank_bands as rank_bands
import ssq_ranking as ranking
import ssq_recommendation_portfolio as recommendation_portfolio
import ssq_recommendation_reporting as recommendation_reporting
import ssq_red_pool as red_pool
import ssq_report_builder
import ssq_report_output as report_output
import ssq_reporting
import ssq_reporting as reporting
import ssq_rule_definitions as rule_definitions
import ssq_rule_execution as rule_execution
import ssq_rule_functions as rule_functions
import ssq_rule_registry as rule_registry
import ssq_rules as rules
import ssq_score_adjustments as score_adjustments
import ssq_score_signals as score_signals
import ssq_scoring as scoring
import ssq_selection as selection
import ssq_strategy_params as strategy_params
import ssq_strategy_validation as strategy_validation
import ssq_training as training
import ssq_workflow as workflow
import ssq_workflow_models as workflow_models
from ssq_config import (
    DEFAULT_PARAMS,
    DEFAULT_STRATEGY_CONFIG,
    StrategyConfig,
)
from ssq_core import (
    BLUE_BALLS,
    LOCAL_TIMEZONE,
    PRIME_RED_BALLS,
    RED_BALLS,
    atomic_write_text,
    infer_next_issue,
    local_now,
    parse_blue_ball,
    parse_blue_balls,
    parse_issue,
    parse_red_balls,
)
from ssq_features import FEATURE_COLUMNS
from ssq_model_contracts import MODEL_TRAINING_PARAMS, BallModelSpec
from ssq_prizes import parse_report_bets
from ssq_rule_models import RecommendationRequest, RuleContext, RuleDefinition
from ssq_rules import select_recommendation_portfolio
from ssq_selection import (
    CandidateGenerationRequest,
    DuplexSelectionRequest,
    generate_candidates,
    rank_duplex_candidates,
)
from ssq_selection_models import RedCandidateSelection
from ssq_workflow_models import PreparedHistory

# ===== from test_ssq_analysis_result.py =====


class AnalysisWorkflowTests(unittest.TestCase):
    def test_default_dependencies_bind_patchable_workflow_boundaries(self):
        dependencies = workflow.default_analysis_dependencies()

        self.assertIs(dependencies.prepare_history, workflow.prepare_history)
        self.assertIs(dependencies.evaluate_history, workflow.evaluate_history)
        self.assertIs(dependencies.train_models, workflow.train_final_models)
        self.assertIs(dependencies.select_current, workflow.select_current_issue)
        self.assertIs(
            dependencies.display_candidates,
            workflow.display_passed_combinations,
        )
        self.assertIs(dependencies.rank_duplex, workflow.rank_duplex_candidates)
        self.assertIs(
            dependencies.build_report_data,
            workflow.build_analysis_report_data,
        )
        self.assertIs(dependencies.now, workflow.local_now)
        self.assertIs(
            dependencies.collect_versions,
            workflow.collect_runtime_versions,
        )
        self.assertIs(dependencies.save_report, workflow.save_analysis_report)

    def test_run_composes_all_stages_through_explicit_dependencies(self):
        options = SimpleNamespace(non_interactive=True)
        history = object()
        history_evaluation = SimpleNamespace(
            loaded_params=SimpleNamespace(values={'weight': 1}),
        )
        selection = SimpleNamespace(
            passed_combos=((1, 2, 3, 4, 5, 6),),
            red_pool=(1, 2, 3, 4, 5, 6, 7),
        )
        current = SimpleNamespace(
            candidate_selection=selection,
            red_scores={1: 1.0},
            rule_context='rule-context',
        )
        generated_at = datetime(2026, 9, 8, tzinfo=UTC)
        prepare_history = Mock(return_value=history)
        evaluate_history = Mock(return_value=history_evaluation)
        train_models = Mock(return_value=('red-workflow_models', 'blue-workflow_models'))
        select_current = Mock(return_value=current)
        display_candidates = Mock()
        rank_duplex = Mock(return_value=['duplex'])
        build_report_data = Mock(return_value='report-data')
        now = Mock(return_value=generated_at)
        collect_versions = Mock(return_value={'python': 'test'})
        save_report = Mock(return_value='report.txt')
        dependencies = replace(
            workflow.default_analysis_dependencies(),
            prepare_history=prepare_history,
            evaluate_history=evaluate_history,
            train_models=train_models,
            select_current=select_current,
            display_candidates=display_candidates,
            rank_duplex=rank_duplex,
            build_report_data=build_report_data,
            now=now,
            collect_versions=collect_versions,
            save_report=save_report,
        )

        with patch('sys.stdout', new_callable=io.StringIO):
            result = workflow.run_analysis(options, dependencies)

        self.assertEqual(result, 'report.txt')
        prepare_history.assert_called_once_with()
        evaluate_history.assert_called_once_with(history, options)
        train_models.assert_called_once_with(history)
        select_current.assert_called_once_with(
            history,
            options,
            history_evaluation.loaded_params.values,
            ('red-workflow_models', 'blue-workflow_models'),
        )
        display_candidates.assert_called_once_with(selection.passed_combos, True)
        duplex_request = rank_duplex.call_args.args[0]
        self.assertIs(duplex_request.passed_combos, selection.passed_combos)
        self.assertIs(duplex_request.red_pool, selection.red_pool)
        self.assertIs(duplex_request.red_scores, current.red_scores)
        self.assertEqual(duplex_request.context, 'rule-context')
        build_report_data.assert_called_once_with(
            history,
            history_evaluation,
            current,
            options,
            ['duplex'],
            generated_at,
            {'python': 'test'},
            workflow.MODEL_TRAINING_PARAMS,
        )
        save_report.assert_called_once_with('report-data')


# ===== from test_ssq_analyzer.py =====


class AnalyzerTests(unittest.TestCase):
  def test_analyzer_reexports_strategy_configuration(self):
    self.assertIs(analyzer.StrategyConfig, config.StrategyConfig)
    self.assertIs(analyzer.parse_cli_options, config.parse_cli_options)
    self.assertEqual(analyzer.DEFAULT_PARAMS, config.DEFAULT_PARAMS)
    self.assertEqual(
        inspect.signature(rules.select_recommendations).parameters['limit'].default,
        config.NUM_RECOMMENDATIONS,
    )

  def test_analyzer_reexports_modeling_functions(self):
    self.assertIs(analyzer.BallModelSpec, modeling.BallModelSpec)
    self.assertIs(analyzer.feature_engineer, modeling.feature_engineer)
    self.assertIs(
        analyzer.run_strategy_and_get_scores,
        modeling.run_strategy_and_get_scores,
    )
    self.assertIs(analyzer.train_models_for_spec, modeling.train_models_for_spec)

  def test_analyzer_reexports_selection_functions(self):
    self.assertIs(
        analyzer.CandidateGenerationRequest,
        selection.CandidateGenerationRequest,
    )
    self.assertIs(analyzer.build_red_pool, selection.build_red_pool)
    self.assertIs(analyzer.generate_candidates, selection.generate_candidates)
    self.assertIs(analyzer.generate_red_candidates, selection.generate_red_candidates)
    self.assertIs(analyzer.DuplexSelectionRequest, selection.DuplexSelectionRequest)
    self.assertIs(analyzer.rank_duplex_candidates, selection.rank_duplex_candidates)
    self.assertIs(analyzer.RecommendationRequest, rules.RecommendationRequest)
    self.assertIs(
        analyzer.select_recommendation_portfolio,
        rules.select_recommendation_portfolio,
    )

  def test_legacy_candidate_api_builds_request_without_losing_options(self):
    strategy = analyzer.StrategyConfig(rejection_lib_size=0)
    context = rules.RuleContext(last_draw={1})
    with patch.object(candidates, 'generate_candidates', return_value='result') as generate:
      result = selection.generate_red_candidates(
          {1: 0.5},
          context,
          {(1, 2, 3, 4, 5, 6)},
          config=strategy,
          mode='middle',
          show_progress=True,
      )

    self.assertEqual(result, 'result')
    request = generate.call_args.args[0]
    self.assertIsInstance(request, selection.CandidateGenerationRequest)
    self.assertEqual(request.red_scores, {1: 0.5})
    self.assertIs(request.context, context)
    self.assertEqual(request.rejection_set, {(1, 2, 3, 4, 5, 6)})
    self.assertIs(request.config, strategy)
    self.assertEqual(request.mode, 'middle')
    self.assertTrue(request.show_progress)

  def test_analyzer_reexports_workflow_entrypoint(self):
    self.assertIs(analyzer.run_analysis, workflow.run_analysis)
    self.assertIs(analyzer.main, workflow.main)

  def test_runtime_versions_cover_model_dependencies(self):
    versions = workflow.collect_runtime_versions()

    self.assertEqual(
        set(versions),
        {'python', 'numpy', 'pandas', 'lightgbm', 'scikit-learn'},
    )
    self.assertTrue(all(isinstance(value, str) and value for value in versions.values()))

  def test_strategy_param_loader_distinguishes_file_states(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'params.json'

      missing = workflow.load_strategy_params(path)
      self.assertFalse(missing.loaded_from_file)
      self.assertEqual(missing.values, analyzer.DEFAULT_PARAMS)

      path.write_text('{invalid', encoding='utf-8')
      with self.assertRaisesRegex(ValueError, '参数文件'):
        workflow.load_strategy_params(path)

      path.write_text(json.dumps(analyzer.DEFAULT_PARAMS), encoding='utf-8')
      loaded = workflow.load_strategy_params(path)
      self.assertTrue(loaded.loaded_from_file)
      self.assertEqual(loaded.values, analyzer.DEFAULT_PARAMS)

  def test_report_declares_the_actual_parseable_recommendation_count(self):
    candidate_selection = selection.RedCandidateSelection(
        (),
        (),
        ((1, 2, 3, 4, 5, 6), (1, 2, 7, 8, 9, 10)),
        ((1, 2, 3, 4, 5, 6), (1, 2, 7, 8, 9, 10)),
    )
    data = SimpleNamespace(
        selection=candidate_selection,
        recommended_blues=[16],
        best_7_reds=[((1, 2, 3, 4, 5, 6, 7), 7)],
    )
    report = '\n'.join(reporting.format_recommendations_report(data))

    with tempfile.NamedTemporaryFile(
        'w', encoding='utf-8', delete=False
    ) as handle:
      handle.write(report)
      path = handle.name
    try:
      single_bets, _ = parse_report_bets(path)
    finally:
      Path(path).unlink()

    self.assertIn('【单式推荐 (2组)】', report)
    self.assertIn('实际任意两注最大重合红球数: 2', report)
    self.assertEqual(len(single_bets), 2)

  def test_report_emits_no_invalid_single_bets_without_a_blue(self):
    candidate_selection = selection.RedCandidateSelection(
        (), (), ((1, 2, 3, 4, 5, 6),), ((1, 2, 3, 4, 5, 6),)
    )
    data = SimpleNamespace(
        selection=candidate_selection,
        recommended_blues=[],
        best_7_reds=[],
    )

    report = '\n'.join(reporting.format_recommendations_report(data))

    self.assertIn('【单式推荐 (0组)】', report)
    self.assertNotIn('组合  1:', report)

  def test_confirmation_input_requires_explicit_y(self):
    for value in ('y', ' Y\n', b'y'):
      with self.subTest(value=value):
        self.assertTrue(workflow.is_confirmation_input(value))
    for value in ('', '\n', 'n', b'\r'):
      with self.subTest(value=value):
        self.assertFalse(workflow.is_confirmation_input(value))

  def test_confirmation_wait_does_not_leak_previous_result(self):
    output = io.StringIO()
    if hasattr(console, 'msvcrt'):
      with (
          patch.object(console.sys, 'stdout', output),
          patch.object(console.msvcrt, 'kbhit', return_value=True),
          patch.object(console.msvcrt, 'getch', return_value=b'y'),
      ):
        self.assertTrue(workflow.get_user_input_with_timeout(1))
      with patch.object(console.sys, 'stdout', output):
        self.assertFalse(workflow.get_user_input_with_timeout(0))
    else:
      with (
          patch.object(console.sys, 'stdout', output),
          patch.object(console.sys, 'stdin', io.StringIO('y\n')),
          patch.object(console.select, 'select', return_value=([object()], [], [])),
      ):
        self.assertTrue(workflow.get_user_input_with_timeout(1))
      with (
          patch.object(console.sys, 'stdout', output),
          patch.object(console.select, 'select', return_value=([], [], [])),
      ):
        self.assertFalse(workflow.get_user_input_with_timeout(0))

  def test_loader_rejects_issue_date_year_mismatch(self):
    content = (
      '期号,日期,红球,蓝球\n'
      '2026001,2025-12-31,"01,02,03,04,05,06",07\n'
    )
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as handle:
      handle.write(content)
      path = handle.name
    try:
      self.assertIsNone(workflow.load_and_preprocess_data(path))
    finally:
      Path(path).unlink()

  def test_loader_rejects_unexpected_schema(self):
    content = (
      'issue,date,reds,blue\n'
      '2026001,2026-01-01,"01,02,03,04,05,06",07\n'
    )
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as handle:
      handle.write(content)
      path = handle.name
    try:
      self.assertIsNone(workflow.load_and_preprocess_data(path))
    finally:
      Path(path).unlink()

  def test_loader_rejects_duplicate_issues(self):
    content = (
      '期号,日期,红球,蓝球\n'
      '2026001,2026-01-01,"01,02,03,04,05,06",07\n'
      '2026001,2026-01-01,"02,03,04,05,06,07",08\n'
    )
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as handle:
      handle.write(content)
      path = handle.name
    try:
      self.assertIsNone(workflow.load_and_preprocess_data(path))
    finally:
      Path(path).unlink()

  def test_loader_parses_and_orders_valid_draws(self):
    content = (
      '蓝球,红球,日期,期号\n'
      '08,"02,03,04,05,06,07",2026-01-04,2026002\n'
      '07,"01,02,03,04,05,06",2026-01-01,2026001\n'
    )
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as handle:
      handle.write(content)
      path = handle.name
    try:
      frame = workflow.load_and_preprocess_data(path)
    finally:
      Path(path).unlink()
    self.assertIsNotNone(frame)
    self.assertEqual(tuple(frame.columns), ('期号', '日期', '红球', '蓝球'))
    self.assertEqual(frame['期号'].tolist(), [2026001, 2026002])
    self.assertEqual(frame.iloc[0]['红球'], [1, 2, 3, 4, 5, 6])
    self.assertEqual(frame.iloc[1]['蓝球'], 8)

  def test_analysis_loader_rejects_future_draws(self):
    content = (
        '期号,日期,红球,蓝球\n'
        '2026001,2026-01-01,"01,02,03,04,05,06",07\n'
    )
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as handle:
      handle.write(content)
      path = handle.name
    try:
      with patch.object(
          workflow,
          'validate_draw_dates_not_future',
          side_effect=ValueError('开奖记录包含未来开奖日期'),
      ):
        self.assertIsNone(workflow.load_and_preprocess_data(path))
    finally:
      Path(path).unlink()

  def test_prime_definition(self):
    self.assertFalse(rules.is_prime(1))
    self.assertTrue(rules.is_prime(2))
    self.assertTrue(rules.is_prime(31))

  def test_feature_engineering_does_not_mutate_source_frame(self):
    source = pandas.DataFrame({
        '红球': [[1, 2, 3, 4, 5, 6]],
        '蓝球': [7],
    })
    result = modeling.feature_engineer(source)
    self.assertEqual(list(source.columns), ['红球', '蓝球'])
    self.assertEqual(
        tuple(column for column in result if column not in source.columns),
        modeling.FEATURE_COLUMNS,
    )

  def test_model_feature_contract_rejects_accidental_columns(self):
    frame = modeling.feature_engineer(pandas.DataFrame({
        '红球': [[1, 2, 3, 4, 5, 6]],
        '蓝球': [7],
        'future_result': [1],
    }))

    self.assertEqual(
        modeling.validate_feature_columns(frame),
        modeling.FEATURE_COLUMNS,
    )
    for feature_columns in (
        (*modeling.FEATURE_COLUMNS, 'future_result'),
        modeling.FEATURE_COLUMNS[:-1],
        (*modeling.FEATURE_COLUMNS, modeling.FEATURE_COLUMNS[-1]),
        tuple(reversed(modeling.FEATURE_COLUMNS)),
    ):
      with self.subTest(feature_columns=feature_columns), self.assertRaises(ValueError):
        modeling.validate_feature_columns(frame, feature_columns)

  def test_omission_uses_row_position_instead_of_index_labels(self):
    history = pandas.DataFrame(
        {'红球': [[1, 2, 3, 4, 5, 6], [2, 7, 8, 9, 10, 11]]},
        index=[100, 500],
    )
    omission = modeling.get_omission(history)
    self.assertEqual(omission[2], 0)
    self.assertEqual(omission[1], 1)
    self.assertEqual(omission[33], 2)


  def test_build_red_pool_mixes_score_bands(self):
    scores = {ball: float(34 - ball) for ball in range(1, 34)}
    pool = selection.build_red_pool(scores)
    self.assertEqual(len(pool), 17)
    self.assertTrue(set(range(1, 5)).issubset(pool))
    self.assertTrue(set(range(30, 34)).issubset(pool))
    self.assertTrue(set(range(13, 22)).issubset(pool))

  def test_build_red_pool_modes(self):
    scores = {ball: float(34 - ball) for ball in range(1, 34)}
    self.assertEqual(selection.build_red_pool(scores, mode="high"), list(range(1, 18)))
    self.assertEqual(selection.build_red_pool(scores, mode="middle"), list(range(9, 26)))
    self.assertEqual(selection.build_red_pool(scores, mode="low"), list(range(17, 34)))
    with self.assertRaises(ValueError):
      selection.build_red_pool(scores, mode="invalid")

  def test_selection_rejects_incomplete_or_non_finite_score_domains(self):
    complete = {ball: float(ball) for ball in range(1, 34)}
    invalid_scores = (
        {ball: score for ball, score in complete.items() if ball != 33},
        {**complete, 34: 1.0},
        {**complete, 1: numpy.nan},
    )

    for scores in invalid_scores:
      with self.subTest(scores=scores), self.assertRaises(ValueError):
        selection.build_red_pool(scores)
      with self.subTest(scores=scores), self.assertRaises(ValueError):
        selection.count_actual_reds_by_rank_band(scores, {1, 2, 3, 4, 5, 6})

  def test_custom_pool_uses_the_same_dynamic_rank_bands_everywhere(self):
    scores = {ball: float(34 - ball) for ball in range(1, 34)}
    custom = analyzer.StrategyConfig(
        pool_size_red=10, high_count=2, low_count=3
    )
    bands = selection.build_rank_bands(custom)

    self.assertEqual(list(bands['high']), [1, 2])
    self.assertEqual(list(bands['middle']), [14, 15, 16, 17, 18])
    self.assertEqual(list(bands['low']), [31, 32, 33])
    self.assertEqual(
        selection.build_red_pool(scores, config=custom),
        [1, 2, 14, 15, 16, 17, 18, 31, 32, 33],
    )
    self.assertEqual(
        selection.build_rank_band_widths(custom),
        {'high': 2, 'middle': 5, 'low': 3, 'other': 23},
    )
    self.assertEqual(
        selection.build_rank_band_labels(custom)['middle'],
        '中段(14-18)',
    )

  def test_candidate_generation_uses_one_selection_pipeline(self):
    scores = {ball: float(ball) for ball in range(1, 34)}
    context = rules.RuleContext(last_draw={1}, previous_draw={2})
    rejected = {(1, 2, 3, 4, 5, 6)}
    config = analyzer.StrategyConfig(
        pool_size_red=7,
        high_count=2,
        low_count=2,
        recommendation_count=2,
        max_shared_red_balls=3,
        rejection_lib_size=1,
    )

    def fake_filter(combo, actual_context, actual_rejection):
      self.assertIs(actual_context, context)
      self.assertIs(actual_rejection, rejected)
      return combo not in actual_rejection

    def fake_select(request):
      return list(request.passed_combos)[:2]

    with (
        patch.object(candidates, 'build_red_pool', return_value=list(range(1, 8))),
        patch.object(candidates, 'passes_red_filters', side_effect=fake_filter) as check,
        patch.object(
            candidates,
            'select_recommendation_portfolio',
            side_effect=fake_select,
        ) as select,
    ):
      result = selection.generate_red_candidates(
          scores, context, rejected, config=config
      )

    self.assertEqual(result.red_pool, tuple(range(1, 8)))
    self.assertEqual(len(result.potential_combos), 7)
    self.assertEqual(len(result.passed_combos), 6)
    self.assertNotIn(next(iter(rejected)), result.passed_combos)
    self.assertEqual(result.recommendations, result.passed_combos[:2])
    self.assertEqual(check.call_count, 7)
    request = select.call_args.args[0]
    self.assertEqual(request.limit, 2)
    self.assertEqual(request.max_shared, 3)
    self.assertIs(request.context, context)

  def test_strategy_config_rejects_incoherent_selection_limits(self):
    with self.assertRaises(ValueError):
      analyzer.StrategyConfig(pool_size_red=5)
    with self.assertRaises(ValueError):
      analyzer.StrategyConfig(pool_size_red=8, high_count=5, low_count=4)
    with self.assertRaises(ValueError):
      analyzer.StrategyConfig(blue_count=17)
    with self.assertRaises(ValueError):
      analyzer.StrategyConfig(recommendation_count=0)
    with self.assertRaises(ValueError):
      analyzer.StrategyConfig(max_shared_red_balls=7)
    with self.assertRaises(ValueError):
      analyzer.StrategyConfig(rejection_lib_size=-1)

  def test_rejection_set_is_reproducible(self):
    first = selection.make_rejection_set(100, random.Random(7))
    second = selection.make_rejection_set(100, random.Random(7))
    self.assertEqual(first, second)
    self.assertEqual(len(first), 100)
    with self.assertRaises(ValueError):
      selection.make_rejection_set(config.TOTAL_RED_COMBINATIONS + 1)

  def test_rejection_set_has_a_stable_seed_fingerprint(self):
    self.assertEqual(
        selection.make_rejection_set(8, random.Random(7)),
        {
            (2, 3, 10, 13, 21, 33),
            (2, 4, 6, 19, 27, 28),
            (2, 4, 13, 15, 19, 31),
            (2, 7, 17, 19, 24, 33),
            (3, 5, 10, 15, 18, 28),
            (3, 6, 8, 14, 31, 33),
            (4, 10, 18, 19, 27, 32),
            (7, 12, 19, 21, 31, 32),
        },
    )

  def test_rejection_seed_is_reproducible_and_issue_specific(self):
    first = selection.rejection_seed_for_issue(42, 2026104)
    self.assertEqual(first, selection.rejection_seed_for_issue(42, "2026104"))
    self.assertNotEqual(first, selection.rejection_seed_for_issue(42, 2026105))
    self.assertNotEqual(first, selection.rejection_seed_for_issue(43, 2026104))


  def test_filter_explanation_includes_prime_rule(self):
    combo = (1, 2, 4, 6, 8, 10)
    failures = rules.explain_filter_failures(
        combo, rules.RuleContext(
            omission_values={n: 0 for n in range(1, 34)}
        )
    )
    self.assertIn("prime_composite_ratio", failures)

  def test_rule_registry_has_one_normalized_score_budget(self):
    self.assertEqual(rules.validate_rule_registry(rules.RED_RULES), rules.RED_RULES)
    self.assertAlmostEqual(
        rules.COMBINATION_SIGNAL_WEIGHT
        + sum(rule.score_weight for rule in rules.RED_RULES),
        1.0,
    )

    duplicate = rules.RuleDefinition(
        rules.RED_RULES[0].name,
        True,
        lambda _combo, _context: True,
    )
    with self.assertRaisesRegex(ValueError, '名称不能重复'):
      rules.validate_rule_registry((*rules.RED_RULES, duplicate))

    ineffective_soft_rule = rules.RuleDefinition(
        'ineffective',
        False,
        lambda _combo, _context: True,
    )
    with self.assertRaisesRegex(ValueError, '软规则'):
      rules.validate_rule_registry((ineffective_soft_rule,), signal_weight=1.0)

    self.assertEqual(
        rules.validate_rule_registry(rule for rule in rules.RED_RULES),
        rules.RED_RULES,
    )

  def test_soft_rule_failure_does_not_reject_combination(self):
    combo = (1, 4, 8, 16, 25, 30)
    self.assertFalse(rules.filter_prime_composite_ratio(combo))
    self.assertTrue(rules.passes_red_filters(
        combo, rules.RuleContext(
            omission_values={n: 0 for n in range(1, 34)}, last_draw={1}
        )
    ))

  def test_rule_context_routes_inputs_by_name(self):
    context = rules.RuleContext(
        omission_values={ball: 16 for ball in range(1, 34)},
        recent_draws=[{1, 2, 3, 4, 20, 21}],
        last_draw={32},
        previous_draw={31},
    )

    failures = rules.explain_filter_failures((1, 2, 3, 4, 5, 6), context)
    self.assertTrue(
        {'recent_overlap', 'all_cold', 'related_numbers'}.issubset(failures)
    )
    diagonal_rule = next(
        rule for rule in rules.RED_RULES if rule.name == 'diagonal_consecutive'
    )
    self.assertFalse(diagonal_rule.evaluator((3, 8, 14, 21, 27, 33), context))

  def test_combination_score_is_deterministic(self):
    scores = {n: n / 33 for n in range(1, 34)}
    combo = (3, 8, 14, 21, 27, 32)
    first = rules.score_red_combination(combo, scores)
    second = rules.score_red_combination(combo, scores)
    self.assertEqual(first, second)

  def test_positive_probability_handles_single_class_models(self):
    features = pandas.DataFrame({'value': range(10)})
    prediction_rows = features.iloc[:2]
    for target, expected in (([0] * 10, 0.0), ([1] * 10, 1.0)):
      model = lightgbm.LGBMClassifier(random_state=42, verbose=-1)
      model.fit(features, target)
      probabilities = modeling.predict_positive_probability(model, prediction_rows)
      self.assertEqual(probabilities.tolist(), [expected, expected])

  def test_ball_models_enable_lightgbm_deterministic_mode(self):
    training = pandas.DataFrame({
        'feature': range(10),
        'outcome': [[1] if index % 2 else [2] for index in range(10)],
    })
    with patch.object(
        modeling.lightgbm,
        'LGBMClassifier',
        wraps=lightgbm.LGBMClassifier,
    ) as classifier:
      modeling.train_ball_models(
          training,
          ('feature',),
          (1,),
          'outcome',
          lambda draw, ball: ball in draw,
      )

    self.assertTrue(classifier.call_args.kwargs['deterministic'])
    self.assertTrue(classifier.call_args.kwargs['force_col_wise'])
    self.assertEqual(classifier.call_args.kwargs['random_state'], 42)
    with self.assertRaises(TypeError):
      modeling.MODEL_TRAINING_PARAMS['random_state'] = 7

  def test_repeated_model_training_produces_identical_ball_scores(self):
    draws = [
        sorted({((index * 5 + offset * 4) % 33) + 1 for offset in range(6)})
        for index in range(40)
    ]
    history = modeling.feature_engineer(pandas.DataFrame({
        '红球': draws,
        '蓝球': [(index % 16) + 1 for index in range(40)],
    }))
    training = history.iloc[5:]

    first_models = modeling.train_prediction_models(
        training,
        modeling.FEATURE_COLUMNS,
    )
    second_models = modeling.train_prediction_models(
        training,
        modeling.FEATURE_COLUMNS,
    )
    first_scores = modeling.run_strategy_and_get_scores(
        history,
        analyzer.DEFAULT_PARAMS,
        *first_models,
        modeling.FEATURE_COLUMNS,
    )
    second_scores = modeling.run_strategy_and_get_scores(
        history,
        analyzer.DEFAULT_PARAMS,
        *second_models,
        modeling.FEATURE_COLUMNS,
    )

    self.assertEqual(first_scores, second_scores)

  def test_positive_probability_uses_the_positive_class_column(self):
    model = lightgbm.LGBMClassifier(random_state=42, verbose=-1)
    features = pandas.DataFrame({'value': range(20)})
    model.fit(features, [0, 1] * 10)
    expected = model.predict_proba(features.iloc[:2])[:, 1]
    actual = modeling.predict_positive_probability(model, features.iloc[:2])
    self.assertTrue(numpy.array_equal(actual, expected))

  def test_positive_probability_rejects_invalid_model_output(self):
    features = pandas.DataFrame({'value': [1]})
    for probabilities in (
        [[0.5, numpy.nan]],
        [[-0.1, 1.1]],
        [[0.5]],
    ):
      model = SimpleNamespace(
          classes_=numpy.array([0, 1]),
          predict_proba=lambda _features, value=probabilities: value,
      )
      with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
        modeling.predict_positive_probability(model, features)

  def test_model_and_score_domains_must_match_all_lottery_balls(self):
    red_models = {ball: object() for ball in range(1, 34)}
    blue_models = {ball: object() for ball in range(1, 17)}
    modeling.validate_model_sets(red_models, blue_models)

    with self.assertRaises(ValueError):
      modeling.validate_model_sets({**red_models, 34: object()}, blue_models)
    with self.assertRaises(ValueError):
      modeling.validate_model_sets(red_models, {1: object()})

    scores = modeling.validate_ball_scores(
        {ball: numpy.float64(ball) for ball in range(1, 34)},
        range(1, 34),
        '红球',
    )
    self.assertEqual(set(scores), set(range(1, 34)))
    self.assertTrue(all(type(value) is float for value in scores.values()))

    with self.assertRaises(ValueError):
      modeling.validate_ball_scores({1: 0.5}, range(1, 34), '红球')
    with self.assertRaises(ValueError):
      modeling.validate_ball_scores(
          {ball: numpy.nan for ball in range(1, 34)}, range(1, 34), '红球'
      )

  def test_strategy_scoring_rejects_empty_history_explicitly(self):
    history = pandas.DataFrame(columns=modeling.FEATURE_COLUMNS)
    red_models = {ball: object() for ball in range(1, 34)}
    blue_models = {ball: object() for ball in range(1, 17)}

    with self.assertRaisesRegex(ValueError, '历史数据不能为空'):
      modeling.run_strategy_and_get_scores(
          history,
          analyzer.DEFAULT_PARAMS,
          red_models,
          blue_models,
          modeling.FEATURE_COLUMNS,
      )

  def test_strategy_scores_with_single_class_training_history(self):
    history = modeling.feature_engineer(pandas.DataFrame({
        '红球': [[1, 2, 3, 4, 5, 6] for _ in range(12)],
        '蓝球': [1 for _ in range(12)],
    }))
    feature_columns = [
        column for column in history.columns if column not in ('红球', '蓝球')
    ]
    red_models, blue_models = modeling.train_prediction_models(
        history.iloc[5:], feature_columns
    )
    red_scores, blue_scores = modeling.run_strategy_and_get_scores(
        history, analyzer.DEFAULT_PARAMS, red_models, blue_models, feature_columns
    )
    self.assertEqual(set(red_scores), set(range(1, 34)))
    self.assertEqual(set(blue_scores), set(range(1, 17)))
    self.assertTrue(all(numpy.isfinite(score) for score in red_scores.values()))
    self.assertTrue(all(numpy.isfinite(score) for score in blue_scores.values()))
    self.assertGreater(red_scores[1], red_scores[7])
    self.assertGreater(blue_scores[1], blue_scores[2])

  def test_rule_registry_preserves_combination_score(self):
    scores = {n: n / 33 for n in range(1, 34)}
    combo = (3, 8, 14, 21, 27, 32)
    last_draw = {2, 7, 13, 20, 26, 31}
    previous_draw = {1, 6, 12, 19, 25, 30}
    signal = rules.score_rank_center_preference(combo, scores)
    expected = (
        0.50 * signal
        + 0.10 * rules.score_odd_even_balance(combo)
        + 0.10 * rules.score_zone_balance(combo)
        + 0.08 * rules.score_prime_balance(combo)
        + 0.08 * rules.score_big_small_balance(combo)
        + 0.05 * float(rules.filter_ac_value(combo))
        + 0.04 * float(rules.filter_modulo3_roads(combo))
        + 0.03 * float(rules.filter_head_tail_range(combo))
        + 0.02 * float(
            rules.filter_diagonal_consecutive(combo, last_draw, previous_draw)
        )
    )
    self.assertAlmostEqual(
        rules.score_red_combination(combo, scores, last_draw, previous_draw),
        expected,
    )

  def test_combination_scoring_receives_the_complete_rule_context(self):
    scores = {ball: float(ball) for ball in range(1, 34)}
    combo = (3, 8, 14, 21, 27, 32)
    context = rules.RuleContext(
        omission_values={1: 7},
        recent_draws=({1, 2, 3},),
        last_draw={4},
        previous_draw={5},
    )
    context_rule = rules.RuleDefinition(
        'context_rule',
        False,
        lambda _combo, _context: True,
        0.25,
        lambda _combo, actual_context: (
            actual_context.omission_values[1] / 7
            if actual_context.recent_draws else 0.0
        ),
    )

    with patch.object(rules, 'RED_RULES', (context_rule,)):
      score = rules.score_red_combination(combo, scores, context=context)

    expected_signal = rules.score_rank_center_preference(combo, scores)
    self.assertAlmostEqual(score, 0.50 * expected_signal + 0.25)

  def test_rank_signal_prefers_center_over_both_extremes(self):
    scores = {n: float(34 - n) for n in range(1, 34)}
    middle = rules.score_rank_center_preference((14, 15, 16, 17, 18, 19), scores)
    high = rules.score_rank_center_preference((1, 2, 3, 4, 5, 6), scores)
    low = rules.score_rank_center_preference((28, 29, 30, 31, 32, 33), scores)
    self.assertGreater(middle, high)
    self.assertGreater(middle, low)
    cached = rules.build_rank_center_scores(scores)
    self.assertEqual(
      middle,
      rules.score_rank_center_preference((14, 15, 16, 17, 18, 19), scores, cached),
    )

  def test_recommendation_portfolio_limits_overlap(self):
    combos = list(combinations(range(1, 13), 6))
    scores = {n: n / 12 for n in range(1, 13)}
    selected = rules.select_recommendations(combos, scores, limit=10, max_shared=4)
    self.assertEqual(len(selected), 10)
    for index, combo in enumerate(selected):
      for other in selected[index + 1:]:
        self.assertLessEqual(len(set(combo) & set(other)), 4)

  def test_legacy_recommendation_api_builds_complete_request(self):
    combos = ((1, 2, 3, 4, 5, 6),)
    scores = {ball: float(ball) for ball in range(1, 34)}
    context = rules.RuleContext(last_draw={7}, previous_draw={8})
    with patch.object(
        rules,
        'select_recommendation_portfolio',
        return_value=['result'],
    ) as select:
      result = rules.select_recommendations(
          combos,
          scores,
          limit=3,
          max_shared=2,
          context=context,
      )

    self.assertEqual(result, ['result'])
    request = select.call_args.args[0]
    self.assertEqual(request.passed_combos, combos)
    self.assertIs(request.red_scores, scores)
    self.assertIs(request.context, context)
    self.assertEqual(request.limit, 3)
    self.assertEqual(request.max_shared, 2)

  def test_duplex_selection_uses_score_to_break_coverage_ties(self):
    pool = list(range(1, 9))
    passed = list(combinations(pool, 6))
    scores = {n: float(34 - n) for n in range(1, 34)}
    ranked = selection.find_best_7_red_combinations(passed, pool, scores)
    self.assertEqual(ranked[0], ((2, 3, 4, 5, 6, 7, 8), 7))

  def test_duplex_quality_scores_only_subtickets_that_passed_hard_rules(self):
    passed = [(1, 2, 3, 4, 5, 6)]
    scores = {ball: float(ball) for ball in range(1, 34)}
    with patch.object(duplex, 'score_combination', return_value=1.0) as score:
      ranked = selection.find_best_7_red_combinations(
          passed,
          range(1, 8),
          scores,
      )

    self.assertEqual(ranked, [((1, 2, 3, 4, 5, 6, 7), 1)])
    self.assertEqual(score.call_count, 1)
    self.assertEqual(score.call_args.args[0], passed[0])

  def test_actual_red_rank_bands_include_unselected_ranks(self):
    scores = {ball: float(34 - ball) for ball in range(1, 34)}
    counts = selection.count_actual_reds_by_rank_band(
        scores, {1, 4, 13, 21, 30, 22}
    )
    self.assertEqual(
        counts,
        {"high": 2, "middle": 2, "low": 1, "other": 1},
    )
    self.assertEqual(sum(counts.values()), 6)

  def test_red_score_adjustments_restore_original_bonuses(self):
    history = pandas.DataFrame({'红球': [
        [1, 2, 3, 4, 5, 6],
        [1, 7, 8, 9, 10, 11],
    ]})
    params = {
        'hot_lookback': 2, 'hot_threshold': 2, 'hot_bonus': 2.0,
        'cold_lookback': 2, 'cold_bonus': 3.0, 'repeat_bonus': 5.0,
    }
    adjusted = modeling.apply_red_score_adjustments(
        {ball: 1.0 for ball in range(1, 34)}, history, params
    )
    self.assertEqual(adjusted[1], 10.0)  # hot and repeated
    self.assertEqual(adjusted[7], 5.0)   # repeated only
    self.assertEqual(adjusted[12], 3.0)  # cold only
    self.assertEqual(adjusted[2], 1.0)   # no adjustment

  def test_strategy_params_are_validated_and_completed(self):
    params = analyzer.validate_strategy_params({})
    self.assertEqual(params['weight_freq'], 0.4)
    with self.assertRaises(ValueError):
      analyzer.validate_strategy_params({'weight_freq': 0.9})
    with self.assertRaises(ValueError):
      analyzer.validate_strategy_params({'decay_factor': 1.1})
    with self.assertRaises(ValueError):
      analyzer.validate_strategy_params({'repeat_bonus': 0})
    with self.assertRaises(ValueError):
      analyzer.validate_strategy_params({'weight_frequency': 1.0})
    with self.assertRaises(TypeError):
      analyzer.validate_strategy_params({'weight_freq': '0.4'})
    with self.assertRaises(TypeError):
      analyzer.validate_strategy_params({'hot_lookback': 10.5})

  def test_bundled_params_match_defaults_and_weights_are_normalized(self):
    with open(analyzer.PARAMS_JSON_PATH, encoding='utf-8') as handle:
      bundled = json.load(handle)

    self.assertEqual(bundled, analyzer.DEFAULT_PARAMS)
    self.assertAlmostEqual(sum(
      bundled[name]
      for name in ('weight_freq', 'weight_omission', 'weight_ml')
    ), 1.0)
    self.assertAlmostEqual(sum(
      bundled[name]
      for name in ('weight_blue_freq', 'weight_blue_ml')
    ), 1.0)

class AntiCrowdingTests(unittest.TestCase):
    def test_selection_preserves_anti_crowding_compatibility_exports(self):
        self.assertIs(
            selection.make_rejection_set,
            anti_crowding.make_rejection_set,
        )
        self.assertIs(
            selection.rejection_seed_for_issue,
            anti_crowding.rejection_seed_for_issue,
        )

    def test_default_size_and_seeded_sample_remain_stable(self):
        self.assertEqual(DEFAULT_STRATEGY_CONFIG.rejection_lib_size, 500_000)
        self.assertEqual(
            anti_crowding.make_rejection_set(8, random.Random(7)),
            {
                (2, 3, 10, 13, 21, 33),
                (2, 4, 6, 19, 27, 28),
                (2, 4, 13, 15, 19, 31),
                (2, 7, 17, 19, 24, 33),
                (3, 5, 10, 15, 18, 28),
                (3, 6, 8, 14, 31, 33),
                (4, 10, 18, 19, 27, 32),
                (7, 12, 19, 21, 31, 32),
            },
        )

    def test_size_and_base_seed_reject_non_integer_values(self):
        for value in (True, 1.5):
            with self.subTest(size=value), self.assertRaises(TypeError):
                anti_crowding.make_rejection_set(value)
            with self.subTest(seed=value), self.assertRaises(TypeError):
                anti_crowding.rejection_seed_for_issue(value, 2026104)


# ===== from test_ssq_backtest_evaluation.py =====


class BonusCalculationTests(unittest.TestCase):
    def test_legacy_bonus_report_api_builds_complete_data_object(self):
        generated_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=LOCAL_TIMEZONE)
        singles = [{'red': [1, 2, 3, 4, 5, 6], 'blue': 7}]
        duplex = {'red': [1, 2, 3, 4, 5, 6, 7], 'blue': [7]}
        with patch(
            'ssq_bonus_reporting.format_bonus_report',
            return_value='report',
        ) as format_report:
            result = bonus_calculation.build_bonus_report(
                'analysis.txt',
                2026001,
                {1, 2, 3, 4, 5, 6},
                7,
                singles,
                duplex,
                generated_at,
            )

        self.assertEqual(result, 'report')
        data = format_report.call_args.args[0]
        self.assertIsInstance(data, bonus_calculation.BonusReportData)
        self.assertEqual(data.report_filepath, 'analysis.txt')
        self.assertEqual(data.target_issue, 2026001)
        self.assertEqual(data.winning_reds, {1, 2, 3, 4, 5, 6})
        self.assertEqual(data.winning_blue, 7)
        self.assertIs(data.single_bets, singles)
        self.assertIs(data.duplex_bet, duplex)
        self.assertIs(data.generated_at, generated_at)

    def test_latest_draw_is_selected_by_validated_issue(self):
        content = (
            '期号,日期,红球,蓝球\n'
            '2026002,2026-01-04,"07,08,09,10,11,12",16\n'
            '2026001,2026-01-01,"01,02,03,04,05,06",07\n'
        )
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as handle:
            handle.write(content)
            path = handle.name
        try:
            latest = bonus_calculation.load_latest_draw(path)
        finally:
            Path(path).unlink()
        self.assertEqual(latest['issue'], 2026002)
        self.assertEqual(latest['red'], {7, 8, 9, 10, 11, 12})
        self.assertEqual(latest['blue'], 16)

    def test_latest_draw_rejects_issue_date_year_mismatch(self):
        content = (
            '期号,日期,红球,蓝球\n'
            '2026001,2025-12-31,"01,02,03,04,05,06",07\n'
        )
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as handle:
            handle.write(content)
            path = handle.name
        try:
            with self.assertRaisesRegex(ValueError, '期号年份'):
                bonus_calculation.load_latest_draw(path)
        finally:
            Path(path).unlink()

    def test_latest_draw_rejects_future_records(self):
        content = (
            '期号,日期,红球,蓝球\n'
            '2026001,2026-01-01,"01,02,03,04,05,06",07\n'
        )
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as handle:
            handle.write(content)
            path = handle.name
        try:
            with patch(
                'ssq_bonus_workflow.validate_draw_dates_not_future',
                side_effect=ValueError('开奖记录包含未来开奖日期'),
            ), self.assertRaisesRegex(ValueError, '未来开奖日期'):
                bonus_calculation.load_latest_draw(path)
        finally:
            Path(path).unlink()

    def test_parse_current_report_format(self):
        content = """【单式推荐 (2组)】
组合 1: 红球 [1,2, 3, 4, 5, 6] 蓝球 [09]
组合 2: 红球 [7, 8, 9, 10, 11, 12] 蓝球 [16]
【7+N 复式推荐 (1组)】
红球: [1, 2, 3, 4, 5, 6, 7]
蓝球: [1, 3, 5, 7, 9, 11, 13]
"""
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as handle:
            handle.write(content)
            path = handle.name
        singles, duplex = bonus_calculation.parse_report_bets(path)
        Path(path).unlink()
        self.assertEqual(len(singles), 2)
        self.assertEqual(duplex['red'], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(duplex['blue'], [1, 3, 5, 7, 9, 11, 13])

    def test_invalid_duplex_numbers_are_rejected(self):
        content = """【单式推荐 (0组)】
【7+N 复式推荐 (1组)】
  红球: [1, 2, 3, 4, 5, 6, 6]
  蓝球: [1, 1]
"""
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as handle:
            handle.write(content)
            path = handle.name
        try:
            with self.assertRaisesRegex(ValueError, '复式投注解析失败'):
                bonus_calculation.parse_report_bets(path)
        finally:
            Path(path).unlink()

    def test_report_rejects_missing_single_bets(self):
        content = """【单式推荐 (2组)】
组合 1: 红球 [1, 2, 3, 4, 5, 6] 蓝球 [09]
【7+N 复式推荐 (1组)】
红球: [1, 2, 3, 4, 5, 6, 7]
蓝球: [1, 3, 5, 7, 9, 11, 13]
"""
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as handle:
            handle.write(content)
            path = handle.name
        try:
            with self.assertRaisesRegex(ValueError, '声明 2 注，实际解析 1 注'):
                bonus_calculation.parse_report_bets(path)
        finally:
            Path(path).unlink()

    def test_report_rejects_duplicate_single_bets(self):
        content = """【单式推荐 (2组)】
组合 1: 红球 [1, 2, 3, 4, 5, 6] 蓝球 [09]
组合 2: 红球 [1, 2, 3, 4, 5, 6] 蓝球 [09]
【7+N 复式推荐 (1组)】
红球: [1, 2, 3, 4, 5, 6, 7]
蓝球: [1, 3, 5, 7, 9, 11, 13]
"""
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as handle:
            handle.write(content)
            path = handle.name
        try:
            with self.assertRaisesRegex(ValueError, '重复单式投注'):
                bonus_calculation.parse_report_bets(path)
        finally:
            Path(path).unlink()

    def test_report_target_issue_metadata_must_be_unique_and_valid(self):
        self.assertEqual(
            bonus_calculation.parse_report_target_issue(
                'Data_Basis_Issue: 2026001\n'
                'Prediction_Target_Issue: 2026002\n'
            ),
            2026002,
        )
        for content in (
            'Data_Basis_Issue: 2026001\n',
            'Prediction_Target_Issue: 999\n',
            (
                'Prediction_Target_Issue: 2026001\n'
                'Prediction_Target_Issue: 2026002\n'
            ),
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                bonus_calculation.parse_report_target_issue(content)

    def test_matching_report_skips_newer_invalid_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory)
            valid = report_dir / 'ssq_analysis_output_20260101_000000.txt'
            invalid = report_dir / 'ssq_analysis_output_20260102_000000.txt'
            valid.write_text(
                'Prediction_Target_Issue: 2026001\n', encoding='utf-8'
            )
            invalid.write_text(
                'Prediction_Target_Issue: 2026001\n'
                'Prediction_Target_Issue: 2026002\n',
                encoding='utf-8',
            )

            result, error = bonus_calculation.find_matching_report(2026001, report_dir)

            self.assertIsNone(error)
            self.assertEqual(Path(result), valid)

    def test_duplex_counts_hit_and_missed_blue_subtickets(self):
        total, breakdown, _ = bonus_calculation.calculate_duplex_prize(
            [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 8],
            {1, 2, 3, 4, 5, 6}, 8
        )
        self.assertEqual(total, 5_625_200)
        self.assertEqual(sum(breakdown.values()), 49)

    def test_duplex_multiplies_nonwinning_blue_options(self):
        total, breakdown, _ = bonus_calculation.calculate_duplex_prize(
            [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7],
            {1, 2, 3, 4, 5, 6}, 8
        )
        self.assertEqual(total, 708_400)
        self.assertEqual(sum(breakdown.values()), 49)

    def test_prize_calculators_reject_invalid_direct_inputs(self):
        with self.assertRaisesRegex(ValueError, '不能重复'):
            bonus_calculation.calculate_single_prize(
                [1, 1, 2, 3, 4, 5], 7, {1, 2, 3, 4, 5, 6}, 7
            )
        with self.assertRaisesRegex(ValueError, '不能重复'):
            bonus_calculation.calculate_duplex_prize(
                [1, 2, 3, 4, 5, 6, 7], [1, 1],
                {1, 2, 3, 4, 5, 6}, 7,
            )

    def test_complete_bonus_workflow_uses_one_generation_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / 'draws.csv'
            csv_path.write_text(
                '期号,日期,红球,蓝球\n'
                '2026001,2026-01-01,"01,02,03,04,05,06",07\n',
                encoding='utf-8',
            )
            analysis_path = root / 'ssq_analysis_output_20251231_000000.txt'
            analysis_path.write_text(
                'Prediction_Target_Issue: 2026001\n'
                '【单式推荐 (1组)】\n'
                '组合  1: 红球 [1, 2, 3, 4, 5, 6] 蓝球 [07]\n'
                '【7+N 复式推荐 (1组)】\n'
                '红球: [1, 2, 3, 4, 5, 6, 7]\n'
                '蓝球: [1, 2, 3, 4, 5, 6, 7]\n',
                encoding='utf-8',
            )
            generated_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=LOCAL_TIMEZONE)

            result = bonus_calculation.run_bonus_check(
                csv_path=csv_path,
                report_dir=root,
                generated_at=generated_at,
            )

            self.assertEqual(
                Path(result).name, 'ssq_bonus_check_20260102_030405.txt'
            )
            report = Path(result).read_text(encoding='utf-8')
            self.assertIn('报告生成时间: 2026-01-02 03:04:05', report)
            self.assertIn('核对开奖期数: 2026001', report)
            self.assertIn('总计参考奖金:', report)


# ===== from test_ssq_candidate_generation.py =====


class CandidateGenerationTests(unittest.TestCase):
    def test_pipeline_uses_injected_boundaries(self):
        request = CandidateGenerationRequest(
            red_scores={ball: float(ball) for ball in range(1, 34)},
            context=RuleContext(),
            rejection_set={(1, 2, 3, 4, 5, 6)},
            config=StrategyConfig(
                pool_size_red=7,
                high_count=2,
                low_count=2,
                recommendation_count=2,
            ),
        )
        validate = Mock()
        build_pool = Mock(return_value=range(1, 8))
        passes_filters = Mock(
            side_effect=lambda combo, *_: combo not in request.rejection_set
        )
        select_portfolio = Mock(
            side_effect=lambda selection: tuple(selection.passed_combos)[:2]
        )
        dependencies = candidate_generation.CandidateGenerationDependencies(
            validate_request=validate,
            build_pool=build_pool,
            passes_filters=passes_filters,
            select_portfolio=select_portfolio,
        )

        result = candidate_generation.generate_candidates(request, dependencies)

        validate.assert_called_once_with(request)
        build_pool.assert_called_once_with(
            request.red_scores,
            config=request.config,
            mode='mixed',
        )
        self.assertEqual(len(result.potential_combos), 7)
        self.assertEqual(len(result.passed_combos), 6)
        self.assertEqual(result.recommendations, result.passed_combos[:2])
        self.assertEqual(passes_filters.call_count, 7)

    def test_production_workflows_do_not_import_candidates_facade(self):
        for filename in (
            'ssq_backtest_evaluation.py',
            'ssq_prediction_workflow.py',
            'ssq_workflow.py',
        ):
            tree = ast.parse(
                Path(__file__).with_name(filename).read_text(encoding='utf-8')
            )
            imports = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            with self.subTest(filename=filename):
                self.assertNotIn('ssq_candidates', imports)


# ===== from test_ssq_candidate_validation.py =====


class CandidateValidationTests(unittest.TestCase):
    def test_candidates_preserves_validation_compatibility_export(self):
        self.assertIs(
            candidates.validate_candidate_generation_request,
            strategy_validation.validate_candidate_generation_request,
        )

    def test_accepts_complete_candidate_request(self):
        request = CandidateGenerationRequest(
            red_scores={ball: float(ball) for ball in range(1, 34)},
            context=RuleContext(),
            rejection_set=set(),
            config=StrategyConfig(),
            mode='mixed',
            show_progress=False,
        )

        self.assertIsNone(strategy_validation.validate_candidate_generation_request(request))

    def test_rejects_non_request_object(self):
        with self.assertRaises(TypeError):
            strategy_validation.validate_candidate_generation_request(object())


# ===== from test_ssq_candidates.py =====


class CandidateModuleTests(unittest.TestCase):
    def test_selection_preserves_candidate_compatibility_exports(self):
        self.assertIs(selection.build_red_pool, candidates.build_red_pool)
        self.assertIs(
            selection.count_actual_reds_by_rank_band,
            candidates.count_actual_reds_by_rank_band,
        )
        self.assertIs(selection.generate_candidates, candidates.generate_candidates)
        self.assertIs(
            selection.generate_red_candidates,
            candidates.generate_red_candidates,
        )
        self.assertIs(
            selection.validate_candidate_generation_request,
            candidates.validate_candidate_generation_request,
        )

    def test_default_mixed_pool_keeps_high_middle_low_counts(self):
        scores = {ball: float(34 - ball) for ball in range(1, 34)}

        self.assertEqual(
            candidates.build_red_pool(scores),
            [1, 2, 3, 4, *range(13, 22), 30, 31, 32, 33],
        )


# ===== from test_ssq_cli.py =====


class CliTests(unittest.TestCase):
    def test_config_preserves_cli_compatibility_exports(self):
        self.assertIs(config.build_argument_parser, ssq_cli.build_argument_parser)
        self.assertIs(config.parse_cli_options, ssq_cli.parse_cli_options)

    def test_parser_reports_model_validation_errors(self):
        with (
            patch('sys.stderr', new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            ssq_cli.parse_cli_options(['--backtest-periods', '-1'])


# ===== from test_ssq_combination_scoring.py =====


class CombinationScoringTests(unittest.TestCase):
    def test_ranking_preserves_scoring_compatibility_exports(self):
        for name in (
            'build_rank_center_scores',
            'score_rank_center_preference',
            'build_combination_score_context',
            'score_combination',
            'score_red_combination',
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(ranking, name), getattr(scoring, name))

    def test_rank_signal_downweights_both_score_extremes(self):
        scores = {number: float(34 - number) for number in range(1, 34)}

        high = scoring.score_rank_center_preference((1, 2, 3, 4, 5, 6), scores)
        middle = scoring.score_rank_center_preference(
            (14, 15, 16, 17, 18, 19),
            scores,
        )
        low = scoring.score_rank_center_preference(
            (28, 29, 30, 31, 32, 33),
            scores,
        )

        self.assertGreater(middle, high)
        self.assertGreater(middle, low)


# ===== from test_ssq_config_models.py =====


class ConfigModelTests(unittest.TestCase):
    def test_config_preserves_model_compatibility_exports(self):
        self.assertIs(config.StrategyConfig, workflow_models.StrategyConfig)
        self.assertIs(config.AnalyzerOptions, workflow_models.AnalyzerOptions)
        self.assertIs(config.LoadedStrategyParams, workflow_models.LoadedStrategyParams)
        self.assertIs(config.DEFAULT_STRATEGY_CONFIG, workflow_models.DEFAULT_STRATEGY_CONFIG)

    def test_default_selection_shape_preserves_strategy(self):
        strategy = workflow_models.DEFAULT_STRATEGY_CONFIG

        self.assertEqual(strategy.pool_size_red, 17)
        self.assertEqual(strategy.high_count, 4)
        self.assertEqual(
            strategy.pool_size_red - strategy.high_count - strategy.low_count,
            9,
        )
        self.assertEqual(strategy.low_count, 4)
        self.assertEqual(strategy.max_shared_red_balls, 4)
        self.assertEqual(strategy.rejection_lib_size, 500_000)


# ===== from test_ssq_console.py =====


class ConsoleDisplayTests(unittest.TestCase):
    def test_small_candidate_set_is_displayed_without_prompt(self):
        combinations = ((1, 2, 3, 4, 5, 6),)
        output = io.StringIO()

        with (
            redirect_stdout(output),
            patch.object(console, 'get_user_input_with_timeout') as prompt,
        ):
            console.display_passed_combinations(combinations, False)

        prompt.assert_not_called()
        self.assertIn('组合数量为 1', output.getvalue())
        self.assertIn('01 02 03 04 05 06', output.getvalue())

    def test_large_candidate_set_is_silent_in_non_interactive_mode(self):
        combinations = ((1, 2, 3, 4, 5, 6),) * 100
        output = io.StringIO()

        with (
            redirect_stdout(output),
            patch.object(console, 'get_user_input_with_timeout') as prompt,
        ):
            console.display_passed_combinations(combinations, True)

        prompt.assert_not_called()
        self.assertEqual(output.getvalue(), '')

    def test_large_candidate_set_is_displayed_after_confirmation(self):
        combinations = ((1, 2, 3, 4, 5, 6),) * 100
        output = io.StringIO()

        with (
            redirect_stdout(output),
            patch.object(
                console,
                'get_user_input_with_timeout',
                return_value=True,
            ) as prompt,
        ):
            console.display_passed_combinations(combinations, False)

        prompt.assert_called_once_with(console.COUNTDOWN_SECONDS)
        self.assertIn('输出所有通过检验的组合', output.getvalue())
        self.assertEqual(output.getvalue().count('01 02 03 04 05 06'), 100)


# ===== from test_ssq_core.py =====


class CoreValidationTests(unittest.TestCase):
    def test_runtime_modules_do_not_depend_on_compatibility_facade(self):
        module_dir = Path(__file__).parent
        offenders = []
        for path in module_dir.glob('ssq_*.py'):
            if path.name == 'ssq_core.py':
                continue
            tree = ast.parse(path.read_text(encoding='utf-8'))
            if any(
                isinstance(node, (ast.Import, ast.ImportFrom))
                and (
                    getattr(node, 'module', None) == 'ssq_core'
                    or any(alias.name == 'ssq_core' for alias in node.names)
                )
                for node in ast.walk(tree)
            ):
                offenders.append(path.name)

        self.assertEqual(offenders, [])

    def test_shared_ball_domains_are_complete(self):
        self.assertEqual(RED_BALLS, tuple(range(1, 34)))
        self.assertEqual(BLUE_BALLS, tuple(range(1, 17)))
        self.assertEqual(
            PRIME_RED_BALLS,
            frozenset({2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}),
        )

    def test_local_now_uses_shanghai_timezone(self):
        current = local_now()

        self.assertEqual(current.tzinfo, LOCAL_TIMEZONE)
        self.assertEqual(current.tzinfo.key, 'Asia/Shanghai')

    def test_parses_and_sorts_valid_red_balls(self):
        self.assertEqual(parse_red_balls('06,01,10,03,08,02'), [1, 2, 3, 6, 8, 10])

    def test_rejects_invalid_red_balls(self):
        for value in ('1,2,3,4,5', '1,1,2,3,4,5', '1,2,3,4,5,34', [1, 2, 3, 4, 5, 6.5]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_red_balls(value)

    def test_rejects_invalid_blue_ball(self):
        self.assertEqual(parse_blue_ball('09'), 9)
        for value in (0, 17, 9.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_blue_ball(value)
        with self.assertRaises(TypeError):
            parse_blue_ball(True)

    def test_parses_and_validates_blue_pool(self):
        self.assertEqual(parse_blue_balls('01, 03,16'), [1, 3, 16])
        for value in ('', '1,1', '1,17'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_blue_balls(value)

    def test_validates_duplex_red_count(self):
        self.assertEqual(
            parse_red_balls('1,2,3,4,5,6,7', expected_count=7),
            [1, 2, 3, 4, 5, 6, 7],
        )

    def test_infers_next_issue_in_same_year(self):
        self.assertEqual(infer_next_issue(2026103, '2026-09-06'), 2026104)

    def test_infers_next_issue_across_year_boundary(self):
        self.assertEqual(infer_next_issue(2025153, '2025-12-30'), 2026001)

    def test_rejects_issue_date_year_mismatch(self):
        with self.assertRaises(ValueError):
            infer_next_issue(2025100, '2026-01-01')

    def test_parses_strict_issue_number(self):
        self.assertEqual(parse_issue('2026001'), 2026001)
        for value in (2026001.5, '2026000', 'invalid'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_issue(value)

    def test_atomic_text_write_replaces_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.txt'
            path.write_text('old', encoding='utf-8')
            atomic_write_text(path, '新内容')
            self.assertEqual(path.read_text(encoding='utf-8'), '新内容')
            self.assertEqual(list(Path(directory).glob('.ssq-*.tmp')), [])


# ===== from test_ssq_duplex.py =====


class DuplexModuleTests(unittest.TestCase):
    def test_selection_preserves_duplex_compatibility_exports(self):
        self.assertIs(
            selection.find_best_7_red_combinations,
            duplex.find_best_7_red_combinations,
        )
        self.assertIs(
            selection.rank_duplex_candidates,
            duplex.rank_duplex_candidates,
        )
        self.assertIs(
            selection.validate_duplex_selection_request,
            duplex.validate_duplex_selection_request,
        )

    def test_equal_duplex_scores_use_stable_number_order(self):
        request = selection.DuplexSelectionRequest(
            passed_combos=((1, 2, 3, 4, 5, 6),),
            red_pool=tuple(range(1, 9)),
        )

        ranked = duplex.rank_duplex_candidates(request)

        self.assertEqual(ranked[0], ((1, 2, 3, 4, 5, 6, 7), 1))
        self.assertEqual(ranked[1], ((1, 2, 3, 4, 5, 6, 8), 1))


# ===== from test_ssq_features.py =====


class FeatureModuleTests(unittest.TestCase):
    def test_modeling_preserves_feature_compatibility_exports(self):
        self.assertIs(modeling.FEATURE_COLUMNS, features.FEATURE_COLUMNS)
        self.assertIs(modeling.feature_engineer, features.feature_engineer)
        self.assertIs(
            modeling.validate_feature_columns,
            features.validate_feature_columns,
        )

    def test_feature_contract_contains_only_lagged_rolling_inputs(self):
        rolling_columns = {
            column for column in features.FEATURE_COLUMNS
            if column.endswith('_ma5')
        }

        self.assertEqual(
            rolling_columns,
            {'red_sum_ma5', 'odd_count_ma5', 'blue_ma5'},
        )


# ===== from test_ssq_history_evaluation.py =====


class HistoryPreparationTests(unittest.TestCase):
    def test_workflow_loader_delegates_explicit_path(self):
        load = Mock(return_value='frame')
        with unittest.mock.patch.object(
            history_preparation,
            'load_and_preprocess_data',
            load,
        ):
            result = workflow.load_and_preprocess_data('history.csv')

        self.assertEqual(result, 'frame')
        load.assert_called_once_with(
            'history.csv',
            workflow.normalize_draw_frame,
            workflow.validate_draw_dates_not_future,
        )

    def test_preparation_rejects_short_history_before_feature_work(self):
        engineer = Mock()
        dependencies = history_preparation.HistoryPreparationDependencies(
            load_data=Mock(return_value=pandas.DataFrame(index=range(49))),
            fingerprint_history=Mock(),
            engineer_features=engineer,
            infer_target_issue=Mock(),
        )

        with self.assertRaisesRegex(SystemExit, '至少需要50期'):
            history_preparation.prepare_history('history.csv', dependencies)

        engineer.assert_not_called()

    def test_preparation_builds_reproducible_history_identity(self):
        frame = pandas.DataFrame({
            '期号': list(range(2026001, 2026051)),
            '日期': [date(2026, 1, 1) for _ in range(50)],
        })
        engineered = frame.assign(red_sum=100)
        dependencies = history_preparation.HistoryPreparationDependencies(
            load_data=Mock(return_value=frame),
            fingerprint_history=Mock(return_value='a' * 64),
            engineer_features=Mock(return_value=engineered),
            infer_target_issue=Mock(return_value=2026052),
        )

        history = history_preparation.prepare_history('history.csv', dependencies)

        self.assertIs(history.frame, engineered)
        self.assertEqual(history.latest_issue, '2026050')
        self.assertEqual(history.target_issue, 2026052)
        self.assertEqual(history.sha256, 'a' * 64)
        dependencies.infer_target_issue.assert_called_once_with(
            '2026050',
            date(2026, 1, 1),
        )


# ===== from test_ssq_model_contracts.py =====


class ModelContractTests(unittest.TestCase):
    def test_training_preserves_model_contract_compatibility_exports(self):
        self.assertIs(training.BallModelSpec, model_contracts.BallModelSpec)
        self.assertIs(
            training.MODEL_TRAINING_PARAMS,
            model_contracts.MODEL_TRAINING_PARAMS,
        )

    def test_model_spec_is_immutable(self):
        spec = model_contracts.BallModelSpec(
            candidates=(1,),
            outcome_column='红球',
            contains_candidate=lambda draw, ball: ball in draw,
        )

        with self.assertRaises((AttributeError, TypeError)):
            spec.outcome_column = '蓝球'

    def test_training_parameters_are_deterministic_and_immutable(self):
        params = model_contracts.MODEL_TRAINING_PARAMS

        self.assertEqual(params['random_state'], 42)
        self.assertTrue(params['deterministic'])
        self.assertTrue(params['force_col_wise'])
        with self.assertRaises(TypeError):
            params['random_state'] = 7


# ===== from test_ssq_model_training.py =====


class ModelTrainingTests(unittest.TestCase):
    def test_model_fit_uses_injected_classifier(self):
        frame = pandas.DataFrame({
            'feature': [1.0, 2.0, 3.0],
            'outcome': [[1], [2], [1]],
        })
        model = Mock()
        classifier = Mock(return_value=model)
        dependencies = model_training.ModelFitDependencies(
            classifier_factory=classifier,
            progress_factory=Mock(),
        )
        spec = BallModelSpec(
            candidates=(1,),
            outcome_column='outcome',
            contains_candidate=lambda draw, ball: ball in draw,
        )

        workflow_models = model_training.train_models_for_spec(
            frame,
            ('feature',),
            spec,
            dependencies,
        )

        self.assertIs(workflow_models[1], model)
        classifier.assert_called_once_with(**MODEL_TRAINING_PARAMS)
        fitted_features, fitted_target = model.fit.call_args.args
        self.assertEqual(fitted_features['feature'].tolist(), [1.0, 2.0])
        self.assertEqual(fitted_target.tolist(), [0.0, 1.0])

    def test_prediction_training_builds_complete_red_and_blue_specs(self):
        validate = Mock(return_value=('feature',))
        train_spec = Mock(side_effect=({'red': 'workflow_models'}, {'blue': 'workflow_models'}))
        dependencies = model_training.PredictionTrainingDependencies(
            validate_features=validate,
            train_spec=train_spec,
        )
        frame = pandas.DataFrame()

        result = model_training.train_prediction_models(
            frame,
            ('feature',),
            show_progress=True,
            dependencies=dependencies,
        )

        self.assertEqual(result, ({'red': 'workflow_models'}, {'blue': 'workflow_models'}))
        red_spec = train_spec.call_args_list[0].args[2]
        blue_spec = train_spec.call_args_list[1].args[2]
        self.assertEqual(red_spec.candidates, RED_BALLS)
        self.assertEqual(blue_spec.candidates, BLUE_BALLS)
        self.assertEqual(red_spec.description, '训练红球模型')
        self.assertEqual(blue_spec.description, '训练蓝球模型')

class ModelValidationTests(unittest.TestCase):
    def test_training_preserves_validation_compatibility_exports(self):
        self.assertIs(training.validate_model_sets, strategy_validation.validate_model_sets)
        self.assertIs(
            training.predict_positive_probability,
            strategy_validation.predict_positive_probability,
        )

    def test_complete_model_domains_are_accepted(self):
        strategy_validation.validate_model_sets(
            {ball: object() for ball in RED_BALLS},
            {ball: object() for ball in BLUE_BALLS},
        )

    def test_probability_requires_one_unique_positive_class(self):
        model = SimpleNamespace(
            classes_=numpy.array([0, 2]),
            predict_proba=lambda _features: [[0.5, 0.5]],
        )

        with self.assertRaises(ValueError):
            strategy_validation.predict_positive_probability(
                model,
                pandas.DataFrame({'feature': [1.0]}),
            )


# ===== from test_ssq_prediction_workflow.py =====


class PredictionWorkflowTests(unittest.TestCase):
    def test_final_training_uses_history_after_feature_warmup(self):
        frame = pandas.DataFrame({'期号': range(30)})
        history = PreparedHistory(frame, ('feature',), '29', 30, 'a' * 64)
        workflow_models = ('red workflow_models', 'blue workflow_models')
        train = Mock(return_value=workflow_models)
        validate = Mock()

        result = prediction_workflow.train_final_models(
            history,
            prediction_workflow.ModelTrainingDependencies(train, validate),
        )

        self.assertEqual(result, workflow_models)
        training_frame = train.call_args.args[0]
        self.assertEqual(training_frame['期号'].tolist(), list(range(5, 30)))
        train.assert_called_once_with(
            training_frame,
            ('feature',),
            show_progress=True,
        )
        validate.assert_called_once_with(*workflow_models)

class RankBandTests(unittest.TestCase):
    def test_default_rank_bands_preserve_candidate_mix(self):
        self.assertEqual(
            rank_bands.RANK_BAND_WIDTHS,
            {'high': 4, 'middle': 9, 'low': 4, 'other': 16},
        )
        self.assertEqual(rank_bands.RANK_BANDS['high'], range(1, 5))
        self.assertEqual(rank_bands.RANK_BANDS['middle'], range(13, 22))
        self.assertEqual(rank_bands.RANK_BANDS['low'], range(30, 34))

    def test_custom_rank_bands_preserve_ranges_and_labels(self):
        config = StrategyConfig(pool_size_red=10, high_count=2, low_count=3)

        self.assertEqual(
            rank_bands.build_rank_bands(config),
            {
                'high': range(1, 3),
                'middle': range(14, 19),
                'low': range(31, 34),
            },
        )
        self.assertEqual(
            rank_bands.build_rank_band_widths(config),
            {'high': 2, 'middle': 5, 'low': 3, 'other': 23},
        )
        self.assertEqual(
            rank_bands.build_rank_band_labels(config),
            {
                'high': '高端(1-2)',
                'middle': '中段(14-18)',
                'low': '低端(31-33)',
                'other': '其他',
            },
        )

    def test_selection_compatibility_exports_match_shared_definitions(self):
        config = StrategyConfig(pool_size_red=10, high_count=2, low_count=3)

        self.assertEqual(selection.RANK_BANDS, rank_bands.RANK_BANDS)
        self.assertEqual(selection.RANK_BAND_WIDTHS, rank_bands.RANK_BAND_WIDTHS)
        self.assertEqual(
            selection.build_rank_bands(config),
            rank_bands.build_rank_bands(config),
        )
        self.assertEqual(
            selection.build_rank_band_widths(config),
            rank_bands.build_rank_band_widths(config),
        )
        self.assertEqual(
            selection.build_rank_band_labels(config),
            rank_bands.build_rank_band_labels(config),
        )


# ===== from test_ssq_ranking.py =====


class RankingTests(unittest.TestCase):
    def test_selection_uses_the_dedicated_ranking_module(self):
        self.assertIs(
            selection.build_combination_score_context,
            ranking.build_combination_score_context,
        )
        self.assertIs(selection.score_combination, ranking.score_combination)
        self.assertIs(
            selection.select_recommendation_portfolio,
            ranking.select_recommendation_portfolio,
        )

    def test_rules_compatibility_wrapper_uses_its_active_registry(self):
        scores = {ball: float(ball) for ball in range(1, 34)}
        combo = (3, 8, 14, 21, 27, 32)
        context = RuleContext(omission_values={1: 7})
        context_rule = RuleDefinition(
            'context_rule',
            False,
            lambda _combo, _context: True,
            0.25,
            lambda _combo, actual_context: actual_context.omission_values[1] / 7,
        )

        with patch.object(rules, 'RED_RULES', (context_rule,)):
            actual = rules.score_red_combination(combo, scores, context=context)

        signal = ranking.score_rank_center_preference(combo, scores)
        self.assertAlmostEqual(actual, 0.50 * signal + 0.25)


# ===== from test_ssq_recommendation_portfolio.py =====


class RecommendationPortfolioTests(unittest.TestCase):
    def test_ranking_preserves_portfolio_compatibility_export(self):
        self.assertIs(
            ranking.select_recommendation_portfolio,
            recommendation_portfolio.select_recommendation_portfolio,
        )

    def test_selection_relaxes_overlap_only_to_fill_requested_count(self):
        combos = (
            (1, 2, 3, 4, 5, 6),
            (1, 2, 3, 4, 5, 7),
            (1, 2, 3, 4, 5, 8),
        )
        request = RecommendationRequest(
            passed_combos=combos,
            red_scores={ball: float(ball) for ball in RED_BALLS},
            context=RuleContext(),
            limit=3,
            max_shared=4,
        )

        selected = recommendation_portfolio.select_recommendation_portfolio(
            request,
            rule_definitions=(),
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual(set(selected), set(combos))
        self.assertEqual(len(set(selected[0]) & set(selected[1])), 5)

    def test_zero_limit_avoids_scoring(self):
        request = RecommendationRequest(
            passed_combos=((1, 2, 3, 4, 5, 6),),
            red_scores={ball: float(ball) for ball in RED_BALLS},
            context=RuleContext(),
            limit=0,
            max_shared=4,
        )

        self.assertEqual(recommendation_portfolio.select_recommendation_portfolio(request), [])


# ===== from test_ssq_recommendation_reporting.py =====


class RecommendationReportingTests(unittest.TestCase):
    def test_reporting_preserves_recommendation_format_compatibility_export(self):
        self.assertIs(
            reporting.format_recommendations_report,
            recommendation_reporting.format_recommendations_report,
        )

    def test_report_formats_overlap_blue_and_duplex_recommendations(self):
        selection = RedCandidateSelection(
            (),
            (),
            ((1, 2, 3, 4, 5, 6), (1, 2, 7, 8, 9, 10)),
            ((1, 2, 3, 4, 5, 6), (1, 2, 7, 8, 9, 10)),
        )
        data = SimpleNamespace(
            selection=selection,
            recommended_blues=[6, 12],
            best_7_reds=[((1, 2, 3, 4, 5, 6, 7), 7)],
        )

        report = '\n'.join(
            recommendation_reporting.format_recommendations_report(data)
        )

        self.assertIn('【单式推荐 (2组)】', report)
        self.assertIn('实际任意两注最大重合红球数: 2', report)
        self.assertIn('蓝球 [06]', report)
        self.assertIn('蓝球: [6, 12]', report)

    def test_report_suppresses_single_bets_without_a_blue(self):
        selection = RedCandidateSelection(
            (), (), ((1, 2, 3, 4, 5, 6),), ((1, 2, 3, 4, 5, 6),)
        )
        data = SimpleNamespace(
            selection=selection,
            recommended_blues=[],
            best_7_reds=[],
        )

        report = '\n'.join(
            recommendation_reporting.format_recommendations_report(data)
        )

        self.assertIn('【单式推荐 (0组)】', report)
        self.assertIn('未能生成足够的单式组合', report)
        self.assertIn('未能生成足够的复式组合', report)
        self.assertNotIn('组合  1:', report)


# ===== from test_ssq_recommendation_validation.py =====


class RecommendationValidationTests(unittest.TestCase):
    def test_ranking_preserves_validation_compatibility_export(self):
        self.assertIs(
            ranking.validate_recommendation_request,
            strategy_validation.validate_recommendation_request,
        )

    def test_request_materializes_generator_and_normalizes_scores(self):
        combos = ((1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 7))
        request = RecommendationRequest(
            passed_combos=(combo for combo in combos),
            red_scores={ball: ball for ball in RED_BALLS},
            context=RuleContext(),
            limit=1,
            max_shared=4,
        )

        actual_combos, scores, limit, max_shared = (
            strategy_validation.validate_recommendation_request(request)
        )

        self.assertEqual(actual_combos, combos)
        self.assertTrue(all(type(value) is float for value in scores.values()))
        self.assertEqual((limit, max_shared), (1, 4))
        self.assertEqual(tuple(request.passed_combos), ())


# ===== from test_ssq_red_pool.py =====


class RedPoolTests(unittest.TestCase):
    def test_candidates_preserves_red_pool_compatibility_exports(self):
        self.assertIs(candidates.build_red_pool, red_pool.build_red_pool)
        self.assertIs(
            candidates.count_actual_reds_by_rank_band,
            red_pool.count_actual_reds_by_rank_band,
        )

    def test_equal_scores_use_ball_number_as_stable_rank_tiebreaker(self):
        scores = {ball: 1.0 for ball in range(1, 34)}

        result = red_pool.build_red_pool(scores)

        self.assertEqual(
            result,
            [1, 2, 3, 4, *range(13, 22), 30, 31, 32, 33],
        )
        self.assertEqual(len(result), DEFAULT_STRATEGY_CONFIG.pool_size_red)

    def test_actual_numbers_are_counted_by_score_rank_band(self):
        scores = {ball: float(34 - ball) for ball in range(1, 34)}

        counts = red_pool.count_actual_reds_by_rank_band(
            scores,
            {1, 13, 21, 30, 33, 25},
        )

        self.assertEqual(
            counts,
            {'high': 1, 'middle': 2, 'low': 2, 'other': 1},
        )


# ===== from test_ssq_report_builder.py =====


class ReportBuilderTests(unittest.TestCase):
    def test_reporting_preserves_report_builder_compatibility_export(self):
        self.assertIs(
            ssq_reporting.build_analysis_report,
            ssq_report_builder.build_analysis_report,
        )

class ReportModelTests(unittest.TestCase):
    def test_reporting_preserves_data_model_compatibility_export(self):
        self.assertIs(reporting.AnalysisReportData, workflow_models.AnalysisReportData)

class ReportOutputTests(unittest.TestCase):
    def test_workflow_preserves_runtime_version_compatibility_export(self):
        self.assertIs(
            workflow.collect_runtime_versions,
            report_output.collect_runtime_versions,
        )

    def test_report_is_built_and_written_with_timestamped_name(self):
        data = SimpleNamespace(
            generated_at=datetime(2026, 9, 7, 12, 34, 56, tzinfo=UTC),
        )
        dependencies = report_output.ReportOutputDependencies(
            build_report=Mock(return_value='analysis report'),
            write_text=report_output.atomic_write_text,
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch('sys.stdout', new_callable=io.StringIO),
        ):
            filepath = report_output.save_analysis_report(
                data,
                directory,
                dependencies,
            )
            content = Path(filepath).read_text(encoding='utf-8')

        self.assertEqual(Path(filepath).name, 'ssq_analysis_output_20260907_123456.txt')
        self.assertEqual(content, 'analysis report')
        dependencies.build_report.assert_called_once_with(data)

    def test_write_failure_becomes_workflow_exit(self):
        data = SimpleNamespace(
            generated_at=datetime(2026, 9, 7, 12, 34, 56, tzinfo=UTC),
        )
        dependencies = report_output.ReportOutputDependencies(
            build_report=Mock(return_value='analysis report'),
            write_text=Mock(side_effect=OSError('disk full')),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch('sys.stdout', new_callable=io.StringIO),
            self.assertRaisesRegex(SystemExit, '写入报告文件失败: disk full'),
        ):
            report_output.save_analysis_report(data, directory, dependencies)


# ===== from test_ssq_rule_definitions.py =====


class RuleDefinitionTests(unittest.TestCase):
    def test_prime_ratio_is_an_ordinary_soft_rule(self):
        rule = next(
            rule for rule in rule_definitions.RED_RULES
            if rule.name == 'prime_composite_ratio'
        )

        self.assertFalse(rule.hard)
        self.assertEqual(rule.score_weight, 0.08)
        self.assertIsNotNone(rule.scorer)


# ===== from test_ssq_rule_diagnostics.py =====


class RuleExecutionTests(unittest.TestCase):
    def setUp(self):
        self.context = RuleContext()
        self.rules = (
            RuleDefinition('positive', True, lambda combo, _context: combo[0] > 0),
            RuleDefinition('even', True, lambda combo, _context: combo[0] % 2 == 0),
            RuleDefinition(
                'soft',
                False,
                lambda combo, _context: combo[0] < 3,
                0.1,
                lambda _combo, _context: 1.0,
            ),
        )

    def test_hard_filtering_ignores_soft_rule_for_admission(self):
        self.assertTrue(
            rule_execution.passes_red_filters((4,), self.context, self.rules)
        )
        self.assertFalse(
            rule_execution.passes_red_filters((3,), self.context, self.rules)
        )

    def test_explanation_includes_soft_rules_and_anti_crowding(self):
        failures = rule_execution.explain_filter_failures(
            (4,),
            self.context,
            self.rules,
            {(4,)},
        )

        self.assertEqual(failures, ['soft', 'anti_crowding'])

class RuleFunctionTests(unittest.TestCase):
    def test_registry_preserves_rule_function_compatibility_exports(self):
        names = (
            'is_prime',
            'calculate_ac_value',
            'filter_prime_composite_ratio',
            'filter_big_small_ratio',
            'score_prime_balance',
            'score_big_small_balance',
        )
        for name in names:
            with self.subTest(name=name):
                self.assertIs(getattr(rule_registry, name), getattr(rule_functions, name))

    def test_prime_ratio_is_evaluated_like_other_balance_ratios(self):
        self.assertFalse(rule_functions.filter_prime_composite_ratio((1, 4, 6, 8, 9, 10)))
        self.assertTrue(rule_functions.filter_prime_composite_ratio((2, 3, 4, 6, 8, 9)))
        self.assertEqual(rule_functions.score_prime_balance((2, 3, 5, 6, 8, 9)), 1.0)


# ===== from test_ssq_rule_models.py =====


class RuleModelTests(unittest.TestCase):
    def test_rules_preserves_model_compatibility_exports(self):
        self.assertIs(rules.RuleContext, workflow_models.RuleContext)
        self.assertIs(rules.RuleDefinition, workflow_models.RuleDefinition)
        self.assertIs(
            rules.CombinationScoreContext,
            workflow_models.CombinationScoreContext,
        )
        self.assertIs(rules.RecommendationRequest, workflow_models.RecommendationRequest)

    def test_rule_context_default_collections_are_not_shared(self):
        first = workflow_models.RuleContext()
        second = workflow_models.RuleContext()

        self.assertIsNot(first.omission_values, second.omission_values)
        self.assertEqual(first.recent_draws, ())
        self.assertIsNone(first.last_draw)
        self.assertIsNone(first.previous_draw)


# ===== from test_ssq_rule_registry.py =====


class RuleRegistryTests(unittest.TestCase):
    def test_prime_ratio_remains_an_ordinary_soft_rule(self):
        prime_rule = next(
            rule for rule in rule_registry.RED_RULES
            if rule.name == 'prime_composite_ratio'
        )

        self.assertFalse(prime_rule.hard)
        self.assertEqual(prime_rule.score_weight, 0.08)
        self.assertIsNotNone(prime_rule.scorer)
        self.assertAlmostEqual(
            rule_registry.COMBINATION_SIGNAL_WEIGHT
            + sum(rule.score_weight for rule in rule_registry.RED_RULES),
            1.0,
        )


# ===== from test_ssq_rule_reporting.py =====


class RuleScoringTests(unittest.TestCase):
    def test_legacy_and_registry_exports_share_score_functions(self):
        for name in (
            'score_zone_balance',
            'score_odd_even_balance',
            'score_prime_balance',
            'score_big_small_balance',
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(rule_functions, name), getattr(scoring, name))
                self.assertIs(getattr(rule_registry, name), getattr(scoring, name))

    def test_prime_balance_remains_a_normal_soft_score(self):
        self.assertEqual(scoring.score_prime_balance((2, 3, 5, 6, 8, 9)), 1.0)
        prime_rule = next(
            rule for rule in rule_registry.RED_RULES
            if rule.name == 'prime_composite_ratio'
        )
        self.assertFalse(prime_rule.hard)
        self.assertEqual(prime_rule.score_weight, 0.08)


# ===== from test_ssq_rule_validation.py =====


class RuleValidationTests(unittest.TestCase):
    def test_registry_preserves_validation_compatibility_exports(self):
        self.assertIs(rule_registry.validate_signal_weight, strategy_validation.validate_signal_weight)
        self.assertIs(rule_registry.validate_rule_names, strategy_validation.validate_rule_names)
        self.assertIs(
            rule_registry.validate_rule_definition,
            strategy_validation.validate_rule_definition,
        )

    def test_validation_rejects_unnormalized_score_budget(self):
        rule = RuleDefinition(
            'soft',
            False,
            lambda _combo, _context: True,
            0.2,
            lambda _combo, _context: 1.0,
        )

        with self.assertRaisesRegex(ValueError, '组合评分总权重必须为 1'):
            strategy_validation.validate_rule_registry((rule,), 0.5)

    def test_validation_accepts_generator_input_once(self):
        actual = strategy_validation.validate_rule_registry(
            (rule for rule in rule_registry.RED_RULES),
            rule_registry.COMBINATION_SIGNAL_WEIGHT,
        )

        self.assertEqual(actual, rule_registry.RED_RULES)


# ===== from test_ssq_score_adjustments.py =====


class ScoreAdjustmentTests(unittest.TestCase):
    def test_scoring_preserves_adjustment_compatibility_export(self):
        self.assertIs(
            scoring.apply_red_score_adjustments,
            score_adjustments.apply_red_score_adjustments,
        )

    def test_hot_cold_and_repeat_multipliers_compose(self):
        history = pandas.DataFrame({'红球': [
            [1, 2, 3, 4, 5, 6],
            [1, 7, 8, 9, 10, 11],
        ]})
        scores = {ball: 1.0 for ball in RED_BALLS}
        params = {
            'hot_lookback': 2,
            'hot_threshold': 2,
            'hot_bonus': 2.0,
            'cold_lookback': 1,
            'cold_bonus': 3.0,
            'repeat_bonus': 5.0,
        }

        result = score_adjustments.apply_red_score_adjustments(
            scores,
            history,
            params,
        )

        self.assertEqual(result[1], 10.0)
        self.assertEqual(result[2], 3.0)
        self.assertEqual(result[7], 5.0)
        self.assertEqual(result[33], 3.0)
        self.assertEqual(scores[1], 1.0)


# ===== from test_ssq_score_signals.py =====


class ScoreSignalTests(unittest.TestCase):
    def test_scoring_preserves_signal_compatibility_exports(self):
        self.assertIs(scoring.get_omission, score_signals.get_omission)
        self.assertIs(
            scoring.get_weighted_frequency,
            score_signals.get_weighted_frequency,
        )

    def test_omission_tracks_last_seen_draw_and_never_seen_balls(self):
        history = pandas.DataFrame({'红球': [
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8],
        ]})

        omission = score_signals.get_omission(history)

        self.assertEqual(omission[1], 2)
        self.assertEqual(omission[2], 1)
        self.assertEqual(omission[3], 0)
        self.assertEqual(omission[33], 3)

    def test_weighted_frequency_favors_more_recent_occurrences(self):
        draws = pandas.Series([[1], [2]])

        frequency = score_signals.get_weighted_frequency(draws, decay_factor=0.5)

        self.assertEqual(frequency[1], 0.5)
        self.assertEqual(frequency[2], 1.0)


# ===== from test_ssq_scoring.py =====


class ScoringModuleTests(unittest.TestCase):
    def test_modeling_preserves_scoring_compatibility_exports(self):
        self.assertIs(modeling.get_omission, scoring.get_omission)
        self.assertIs(
            modeling.get_weighted_frequency,
            scoring.get_weighted_frequency,
        )
        self.assertIs(
            modeling.apply_red_score_adjustments,
            scoring.apply_red_score_adjustments,
        )
        self.assertIs(
            modeling.run_strategy_and_get_scores,
            scoring.run_strategy_and_get_scores,
        )

    def test_signal_fusion_preserves_declared_red_and_blue_weights(self):
        history = pandas.DataFrame({
            '红球': [[1, 2, 3, 4, 5, 6]],
            '蓝球': [1],
            **{column: [0.0] for column in FEATURE_COLUMNS},
        })
        red_frequency = pandas.Series({ball: 1.0 for ball in RED_BALLS})
        blue_frequency = pandas.Series({ball: 1.0 for ball in BLUE_BALLS})
        red_models = {ball: object() for ball in RED_BALLS}
        blue_models = {ball: object() for ball in BLUE_BALLS}

        with (
            patch.object(scoring, 'validate_model_sets'),
            patch.object(
                scoring,
                'get_weighted_frequency',
                side_effect=(red_frequency, blue_frequency),
            ),
            patch.object(
                scoring,
                'get_omission',
                return_value={ball: 1 for ball in RED_BALLS},
            ),
            patch.object(
                scoring,
                'predict_positive_probability',
                return_value=numpy.array([1.0]),
            ),
            patch.object(
                scoring,
                'apply_red_score_adjustments',
                side_effect=lambda scores, _history, _params: scores,
            ),
        ):
            red_scores, blue_scores = scoring.run_strategy_and_get_scores(
                history,
                DEFAULT_PARAMS,
                red_models,
                blue_models,
                FEATURE_COLUMNS,
            )

        self.assertTrue(all(score == 1.0 for score in red_scores.values()))
        self.assertTrue(all(score == 1.0 for score in blue_scores.values()))
        self.assertEqual(
            (
                DEFAULT_PARAMS['weight_freq'],
                DEFAULT_PARAMS['weight_omission'],
                DEFAULT_PARAMS['weight_ml'],
            ),
            (0.4, 0.5, 0.1),
        )
        self.assertEqual(
            (
                DEFAULT_PARAMS['weight_blue_freq'],
                DEFAULT_PARAMS['weight_blue_ml'],
            ),
            (0.6, 0.4),
        )


# ===== from test_ssq_selection_models.py =====


class SelectionModelTests(unittest.TestCase):
    def test_selection_preserves_model_compatibility_exports(self):
        self.assertIs(
            selection.RedCandidateSelection,
            workflow_models.RedCandidateSelection,
        )
        self.assertIs(
            selection.CandidateGenerationRequest,
            workflow_models.CandidateGenerationRequest,
        )
        self.assertIs(
            selection.DuplexSelectionRequest,
            workflow_models.DuplexSelectionRequest,
        )

    def test_duplex_context_defaults_are_not_shared(self):
        first = workflow_models.DuplexSelectionRequest((), tuple(range(1, 8)))
        second = workflow_models.DuplexSelectionRequest((), tuple(range(1, 8)))

        self.assertIsNot(first.context, second.context)


# ===== from test_ssq_selection_requests.py =====


def red_scores():
    return {ball: float(ball) for ball in RED_BALLS}


class SelectionRequestValidationTests(unittest.TestCase):
    def test_candidate_request_rejects_invalid_controls(self):
        defaults = {
            'red_scores': red_scores(),
            'context': RuleContext(),
            'rejection_set': set(),
        }
        invalid = (
            {'context': object()},
            {'config': object()},
            {'mode': 'unknown'},
            {'show_progress': 1},
            {'rejection_set': iter(())},
        )

        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(
                (TypeError, ValueError)
            ):
                generate_candidates(CandidateGenerationRequest(
                    **{**defaults, **changes}
                ))

    def test_duplex_request_rejects_inconsistent_collections(self):
        defaults = {
            'passed_combos': ((1, 2, 3, 4, 5, 6),),
            'red_pool': tuple(range(1, 8)),
        }
        invalid = (
            {'context': object()},
            {'red_pool': (1, 2, 3, 4, 5, 6)},
            {'red_pool': (1, 2, 3, 4, 5, 6, 34)},
            {'passed_combos': iter(defaults['passed_combos'])},
            {'passed_combos': ((1, 2, 3, 4, 5, 8),)},
            {'passed_combos': defaults['passed_combos'] * 2},
            {'red_scores': {}},
        )

        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(
                (TypeError, ValueError)
            ):
                rank_duplex_candidates(DuplexSelectionRequest(
                    **{**defaults, **changes}
                ))

    def test_recommendation_request_normalizes_generator_once(self):
        combos = ((1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 7))
        request = RecommendationRequest(
            passed_combos=(combo for combo in combos),
            red_scores=red_scores(),
            context=RuleContext(),
            limit=1,
            max_shared=4,
        )

        result = select_recommendation_portfolio(request)

        self.assertEqual(len(result), 1)
        self.assertIn(result[0], combos)
        self.assertEqual(tuple(request.passed_combos), ())

    def test_recommendation_request_rejects_invalid_controls(self):
        defaults = {
            'passed_combos': ((1, 2, 3, 4, 5, 6),),
            'red_scores': red_scores(),
            'context': RuleContext(),
        }
        invalid = (
            {'context': object()},
            {'limit': True},
            {'limit': -1},
            {'max_shared': 4.5},
            {'passed_combos': defaults['passed_combos'] * 2},
            {'passed_combos': ((1, 2, 3, 4, 5, 34),)},
            {'red_scores': {}},
        )

        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(
                (TypeError, ValueError)
            ):
                select_recommendation_portfolio(RecommendationRequest(
                    **{**defaults, **changes}
                ))


# ===== from test_ssq_strategy_params.py =====


class StrategyParamTests(unittest.TestCase):
    def test_missing_file_returns_validated_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missing.json'

            loaded = strategy_params.load_strategy_params(path)

        self.assertFalse(loaded.loaded_from_file)
        self.assertEqual(loaded.values, DEFAULT_PARAMS)

    def test_valid_and_invalid_json_are_distinguished(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'params.json'
            path.write_text('{invalid', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '参数文件'):
                strategy_params.load_strategy_params(path)

            path.write_text(json.dumps(DEFAULT_PARAMS), encoding='utf-8')
            loaded = strategy_params.load_strategy_params(path)

        self.assertTrue(loaded.loaded_from_file)
        self.assertEqual(loaded.values, DEFAULT_PARAMS)

    def test_workflow_wrapper_delegates_explicit_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missing.json'

            loaded = workflow.load_strategy_params(path)

        self.assertFalse(loaded.loaded_from_file)


# ===== from test_ssq_strategy_validation.py =====


class StrategyValidationTests(unittest.TestCase):
    def test_config_preserves_validation_compatibility_exports(self):
        for name in (
            'normalize_integer_param',
            'normalize_float_param',
            'validate_weight_group',
            'validate_param_ranges',
            'validate_strategy_params',
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(config, name), getattr(strategy_validation, name))

    def test_default_scoring_weights_preserve_strategy(self):
        params = strategy_validation.validate_strategy_params({})

        self.assertEqual(
            [params[name] for name in (
                'weight_freq', 'weight_omission', 'weight_ml',
            )],
            [0.4, 0.5, 0.1],
        )
        self.assertEqual(
            [params[name] for name in ('weight_blue_freq', 'weight_blue_ml')],
            [0.6, 0.4],
        )


# ===== from test_ssq_training.py =====


class TrainingModuleTests(unittest.TestCase):
    def test_modeling_preserves_training_compatibility_exports(self):
        self.assertIs(modeling.BallModelSpec, training.BallModelSpec)
        self.assertIs(
            modeling.MODEL_TRAINING_PARAMS,
            training.MODEL_TRAINING_PARAMS,
        )
        self.assertIs(modeling.train_models_for_spec, training.train_models_for_spec)
        self.assertIs(
            modeling.train_prediction_models,
            training.train_prediction_models,
        )
        self.assertIs(
            modeling.predict_positive_probability,
            training.predict_positive_probability,
        )
        self.assertIs(modeling.lightgbm, training.lightgbm)

    def test_training_parameters_remain_deterministic_and_immutable(self):
        self.assertEqual(training.MODEL_TRAINING_PARAMS['random_state'], 42)
        self.assertTrue(training.MODEL_TRAINING_PARAMS['deterministic'])
        with self.assertRaises(TypeError):
            training.MODEL_TRAINING_PARAMS['random_state'] = 7


# ===== from test_ssq_workflow_models.py =====


class WorkflowModelTests(unittest.TestCase):
    def test_workflow_preserves_model_compatibility_exports(self):
        for name in ('PreparedHistory', 'HistoricalEvaluation', 'CurrentSelection'):
            with self.subTest(name=name):
                self.assertIs(getattr(workflow, name), getattr(workflow_models, name))

