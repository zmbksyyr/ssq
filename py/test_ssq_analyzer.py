import inspect
import io
import json
import random
import sys
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ssq_analyzer as analyzer
import ssq_backtest_metrics as backtest_metrics
import ssq_backtesting as backtesting
import ssq_config as config
import ssq_modeling as modeling
import ssq_reporting as reporting
import ssq_rule_auditing as auditing
import ssq_rules as rules
import ssq_selection as selection
import ssq_workflow as workflow
from ssq_prizes import parse_report_bets


class AnalyzerTests(unittest.TestCase):
  def test_cli_options_build_runtime_configuration(self):
    defaults = config.parse_cli_options([])
    self.assertEqual(defaults.rejection_size, 500_000)
    self.assertEqual(defaults.backtest_pool_modes, ('mixed',))
    self.assertEqual(defaults.strategy_config.rejection_lib_size, 500_000)
    self.assertEqual(defaults.strategy_config.max_shared_red_balls, 4)

    options = config.parse_cli_options([
        '--backtest-periods', '12',
        '--rejection-size', '34',
        '--seed', '7',
        '--pool-mode', 'middle',
        '--compare-pools',
        '--non-interactive',
        '--rule-audit-periods', '50',
    ])
    self.assertEqual(options.backtest_periods, 12)
    self.assertEqual(options.rule_audit_periods, 50)
    self.assertEqual(options.pool_mode, 'middle')
    self.assertEqual(options.backtest_pool_modes, config.RED_POOL_MODES)
    self.assertTrue(options.non_interactive)
    self.assertEqual(options.strategy_config.random_seed, 7)
    self.assertEqual(options.strategy_config.rejection_lib_size, 34)

  def test_cli_options_reject_invalid_ranges(self):
    for argv in (
        ['--backtest-periods', '-1'],
        ['--rule-audit-periods', '-1'],
        ['--rejection-size', str(config.TOTAL_RED_COMBINATIONS + 1)],
    ):
      with (
          self.subTest(argv=argv),
          patch.object(sys, 'stderr', io.StringIO()),
          self.assertRaises(SystemExit),
      ):
        config.parse_cli_options(argv)

    with self.assertRaises(ValueError):
      config.AnalyzerOptions(backtest_periods=-1)
    with self.assertRaises(ValueError):
      config.AnalyzerOptions(pool_mode='invalid')

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
    with patch.object(selection, 'generate_candidates', return_value='result') as generate:
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

  def test_analyzer_reexports_backtesting_functions(self):
    self.assertIs(analyzer.BacktestResult, backtesting.BacktestResult)
    self.assertIs(backtesting.BacktestResult, backtest_metrics.BacktestResult)
    self.assertIs(
        backtesting.BacktestAccumulator,
        backtest_metrics.BacktestAccumulator,
    )
    self.assertIs(analyzer.BacktestIssue, backtesting.BacktestIssue)
    self.assertIs(analyzer.BacktestRequest, backtesting.BacktestRequest)
    self.assertIs(
        analyzer.BacktestSelectionInputs,
        backtesting.BacktestSelectionInputs,
    )
    self.assertIs(analyzer.run_backtest, backtesting.run_backtest)
    self.assertIs(analyzer.run_full_backtest, backtesting.run_full_backtest)

  def test_analyzer_reexports_workflow_entrypoint(self):
    self.assertIs(analyzer.run_analysis, workflow.run_analysis)
    self.assertIs(analyzer.main, workflow.main)

  def test_workflow_main_passes_parsed_options(self):
    with patch.object(workflow, 'run_analysis', return_value='report.txt') as run:
      result = workflow.main(['--backtest-periods', '3', '--non-interactive'])
    self.assertEqual(result, 'report.txt')
    options = run.call_args.args[0]
    self.assertEqual(options.backtest_periods, 3)
    self.assertTrue(options.non_interactive)

  def test_analysis_workflow_maps_stage_results_into_report(self):
    options = config.AnalyzerOptions(
        backtest_periods=0,
        rejection_size=0,
        non_interactive=True,
    )
    history = workflow.PreparedHistory(
        frame=pd.DataFrame(),
        feature_columns=('feature',),
        latest_issue='2026103',
        target_issue=2026104,
        sha256='a' * 64,
    )
    loaded_params = config.LoadedStrategyParams({'param': 1}, True)
    evaluation = workflow.HistoricalEvaluation(
        loaded_params=loaded_params,
        rule_coverage={'rule': 'coverage'},
        hard_pipeline_coverage={'hard': 'coverage'},
        backtests={'mixed': 'backtest'},
        selected_backtest='backtest',
    )
    candidate_selection = SimpleNamespace(
        passed_combos=((1, 2, 3, 4, 5, 6),),
        red_pool=(1, 2, 3, 4, 5, 6, 7),
    )
    current = workflow.CurrentSelection(
        red_scores={number: float(number) for number in range(1, 8)},
        recommended_blues=[16],
        rejection_seed=123,
        rule_context='context',
        candidate_selection=candidate_selection,
        pipeline_stats=[{'stage': 'hard'}],
    )
    generated_at = datetime(2026, 9, 7, 12, 34, 56, tzinfo=timezone.utc)

    with (
        patch.object(workflow, 'prepare_history', return_value=history),
        patch.object(workflow, 'evaluate_history', return_value=evaluation),
        patch.object(workflow, 'train_final_models', return_value=('red', 'blue')),
        patch.object(workflow, 'select_current_issue', return_value=current),
        patch.object(workflow, 'display_passed_combinations') as display,
        patch.object(
            workflow,
            'rank_duplex_candidates',
            return_value=['best'],
        ) as rank_duplex,
        patch.object(workflow, 'local_now', return_value=generated_at),
        patch.object(workflow, 'collect_runtime_versions', return_value={'python': 'test'}),
        patch.object(workflow, 'save_analysis_report', return_value='report.txt') as save,
        patch('sys.stdout', new_callable=io.StringIO),
    ):
      result = workflow.run_analysis(options)

    self.assertEqual(result, 'report.txt')
    display.assert_called_once_with(candidate_selection.passed_combos, True)
    duplex_request = rank_duplex.call_args.args[0]
    self.assertIs(duplex_request.passed_combos, candidate_selection.passed_combos)
    self.assertIs(duplex_request.red_pool, candidate_selection.red_pool)
    self.assertIs(duplex_request.red_scores, current.red_scores)
    self.assertIs(duplex_request.context, current.rule_context)
    report_data = save.call_args.args[0]
    self.assertEqual(report_data.latest_issue, history.latest_issue)
    self.assertEqual(report_data.target_issue, history.target_issue)
    self.assertIs(report_data.backtest, evaluation.selected_backtest)
    self.assertIs(report_data.selection, candidate_selection)
    self.assertEqual(report_data.recommended_blues, [16])
    self.assertEqual(report_data.rejection_seed, 123)
    self.assertEqual(report_data.history_sha256, 'a' * 64)
    self.assertEqual(report_data.runtime_versions, {'python': 'test'})
    self.assertEqual(report_data.model_features, ('feature',))
    self.assertIs(report_data.model_training_params, modeling.MODEL_TRAINING_PARAMS)

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

  def test_analysis_report_builder_preserves_section_contracts(self):
    backtest = backtesting.BacktestResult(0, 0, 0, 0, 0, Counter())
    candidate_selection = selection.RedCandidateSelection((), (), (), ())
    coverage = {
        name: {'passed': 0, 'total': 0, 'rate': 0.0}
        for name in analyzer.FILTER_NAMES
    }
    data = analyzer.AnalysisReportData(
        latest_issue='2026103',
        target_issue=2026104,
        generated_at=datetime(2026, 9, 7, 12, 34, 56, tzinfo=timezone.utc),
        params_loaded=False,
        params=analyzer.DEFAULT_PARAMS,
        config=analyzer.DEFAULT_STRATEGY_CONFIG,
        rejection_seed=123,
        backtest=backtest,
        backtests={'mixed': backtest},
        pool_mode='mixed',
        rank_band_widths=analyzer.RANK_BAND_WIDTHS,
        rank_band_labels=selection.build_rank_band_labels(),
        pipeline_stats=[],
        rule_coverage=coverage,
        hard_pipeline_coverage={
            'stages': [], 'passed': 0, 'total': 0, 'rate': 0.0,
        },
        rule_audit_periods=200,
        selection=candidate_selection,
        recommended_blues=[],
        best_7_reds=[],
        runtime_versions={
            'python': '3.11.0',
            'lightgbm': '4.7.0',
        },
        history_sha256='a' * 64,
        model_features=('red_sum', 'red_span'),
        model_training_params={
            'random_state': 42,
            'deterministic': True,
        },
    )

    report = analyzer.build_analysis_report(data)

    self.assertIn('Data_Basis_Issue: 2026103', report)
    self.assertIn('报告生成时间: 2026-09-07 12:34:56', report)
    self.assertIn(f"Data_History_SHA256: {'a' * 64}", report)
    self.assertIn('Runtime_python: 3.11.0', report)
    self.assertIn('Runtime_lightgbm: 4.7.0', report)
    self.assertIn('Model_Features: red_sum,red_span', report)
    self.assertIn('Model_LightGBM_random_state: 42', report)
    self.assertIn('Model_LightGBM_deterministic: True', report)
    self.assertIn('模式: 使用内置的默认参数', report)
    self.assertIn('mixed_pool_bands    : 4 high + 9 middle + 4 low', report)
    self.assertIn('max_shared_red_balls: 4', report)
    self.assertIn('[软] prime_composite_ratio', report)
    self.assertIn('未能生成足够的单式组合', report)
    self.assertIn('未能生成足够的复式组合', report)

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

  def test_rule_report_discloses_truncated_audit_windows(self):
    coverage = {
        name: {'passed': 10, 'total': 12, 'rate': 10 / 12}
        for name in rules.FILTER_NAMES
    }
    data = SimpleNamespace(
        pipeline_stats=[],
        rule_coverage=coverage,
        hard_pipeline_coverage={
            'stages': [],
            'passed': 10,
            'total': 12,
            'rate': 10 / 12,
        },
        rule_audit_periods=200,
    )

    report = '\n'.join(reporting.format_rule_audit_report(data))

    self.assertIn('实际 12 期，请求 200 期，逐条独立统计', report)
    self.assertIn('实际 12 期，请求 200 期，不含随机撞号', report)

  def test_confirmation_input_requires_explicit_y(self):
    for value in ('y', ' Y\n', b'y'):
      with self.subTest(value=value):
        self.assertTrue(workflow.is_confirmation_input(value))
    for value in ('', '\n', 'n', b'\r'):
      with self.subTest(value=value):
        self.assertFalse(workflow.is_confirmation_input(value))

  def test_confirmation_wait_does_not_leak_previous_result(self):
    output = io.StringIO()
    if hasattr(workflow, 'msvcrt'):
      with (
          patch.object(workflow.sys, 'stdout', output),
          patch.object(workflow.msvcrt, 'kbhit', return_value=True),
          patch.object(workflow.msvcrt, 'getch', return_value=b'y'),
      ):
        self.assertTrue(workflow.get_user_input_with_timeout(1))
      with patch.object(workflow.sys, 'stdout', output):
        self.assertFalse(workflow.get_user_input_with_timeout(0))
    else:
      with (
          patch.object(workflow.sys, 'stdout', output),
          patch.object(workflow.sys, 'stdin', io.StringIO('y\n')),
          patch.object(workflow.select, 'select', return_value=([object()], [], [])),
      ):
        self.assertTrue(workflow.get_user_input_with_timeout(1))
      with (
          patch.object(workflow.sys, 'stdout', output),
          patch.object(workflow.select, 'select', return_value=([], [], [])),
      ):
        self.assertFalse(workflow.get_user_input_with_timeout(0))

  def test_default_backtest_uses_stable_window(self):
    self.assertEqual(analyzer.BACKTEST_PERIODS, 200)

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
    source = pd.DataFrame({
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
    frame = modeling.feature_engineer(pd.DataFrame({
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
    history = pd.DataFrame(
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
        {**complete, 1: np.nan},
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
        patch.object(selection, 'build_red_pool', return_value=list(range(1, 8))),
        patch.object(selection, 'passes_red_filters', side_effect=fake_filter) as check,
        patch.object(
            selection,
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

  def test_runtime_configs_reject_non_integer_and_non_boolean_values(self):
    integer_fields = {
        'pool_size_red': 17.5,
        'high_count': True,
        'low_count': 4.5,
        'blue_count': 7.5,
        'recommendation_count': False,
        'max_shared_red_balls': 4.5,
        'rejection_lib_size': 1.5,
        'random_seed': 42.5,
    }
    for name, value in integer_fields.items():
      with self.subTest(name=name), self.assertRaises(TypeError):
        analyzer.StrategyConfig(**{name: value})

    for kwargs in (
        {'backtest_periods': 1.5},
        {'rejection_size': True},
        {'seed': 2.5},
        {'rule_audit_periods': False},
        {'compare_pools': 1},
        {'non_interactive': 0},
    ):
      with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
        config.AnalyzerOptions(**kwargs)

  def test_runtime_configs_normalize_numpy_integers(self):
    strategy = analyzer.StrategyConfig(
        pool_size_red=np.int64(17), random_seed=np.int64(42)
    )
    options = config.AnalyzerOptions(backtest_periods=np.int64(3))

    self.assertIs(type(strategy.pool_size_red), int)
    self.assertIs(type(strategy.random_seed), int)
    self.assertIs(type(options.backtest_periods), int)


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

  def test_all_rules_use_one_unique_registry(self):
    names = [rule.name for rule in rules.RED_RULES]
    self.assertEqual(len(names), len(set(names)))
    self.assertEqual(tuple(names), rules.FILTER_NAMES)
    prime_rule = next(rule for rule in rules.RED_RULES if rule.name == 'prime_composite_ratio')
    self.assertFalse(prime_rule.hard)
    self.assertTrue(all(rule.scorer is not None for rule in rules.RED_RULES if not rule.hard))
    self.assertTrue(all(rule.score_weight > 0 for rule in rules.RED_RULES if not rule.hard))
    self.assertAlmostEqual(
        rules.COMBINATION_SIGNAL_WEIGHT
        + sum(rule.score_weight for rule in rules.RED_RULES),
        1.0,
    )

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

  def test_filter_pipeline_stats_are_incremental(self):
    stats = rules.filter_pipeline_stats(
        [(3, 8, 13, 20, 27, 31), (1, 2, 3, 4, 5, 6)],
        rules.RuleContext(omission_values={n: 0 for n in range(1, 34)}),
    )
    self.assertGreater(len(stats), 10)
    for item in stats:
      self.assertGreaterEqual(item["before"], item["remaining"])

  def test_combination_score_is_deterministic(self):
    scores = {n: n / 33 for n in range(1, 34)}
    combo = (3, 8, 14, 21, 27, 32)
    first = rules.score_red_combination(combo, scores)
    second = rules.score_red_combination(combo, scores)
    self.assertEqual(first, second)

  def test_positive_probability_handles_single_class_models(self):
    features = pd.DataFrame({'value': range(10)})
    prediction_rows = features.iloc[:2]
    for target, expected in (([0] * 10, 0.0), ([1] * 10, 1.0)):
      model = lgb.LGBMClassifier(random_state=42, verbose=-1)
      model.fit(features, target)
      probabilities = modeling.predict_positive_probability(model, prediction_rows)
      self.assertEqual(probabilities.tolist(), [expected, expected])

  def test_ball_models_enable_lightgbm_deterministic_mode(self):
    training = pd.DataFrame({
        'feature': range(10),
        'outcome': [[1] if index % 2 else [2] for index in range(10)],
    })
    with patch.object(
        modeling.lgb,
        'LGBMClassifier',
        wraps=lgb.LGBMClassifier,
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
    history = modeling.feature_engineer(pd.DataFrame({
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
    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    features = pd.DataFrame({'value': range(20)})
    model.fit(features, [0, 1] * 10)
    expected = model.predict_proba(features.iloc[:2])[:, 1]
    actual = modeling.predict_positive_probability(model, features.iloc[:2])
    self.assertTrue(np.array_equal(actual, expected))

  def test_positive_probability_rejects_invalid_model_output(self):
    features = pd.DataFrame({'value': [1]})
    for probabilities in (
        [[0.5, np.nan]],
        [[-0.1, 1.1]],
        [[0.5]],
    ):
      model = SimpleNamespace(
          classes_=np.array([0, 1]),
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
        {ball: np.float64(ball) for ball in range(1, 34)},
        range(1, 34),
        '红球',
    )
    self.assertEqual(set(scores), set(range(1, 34)))
    self.assertTrue(all(type(value) is float for value in scores.values()))

    with self.assertRaises(ValueError):
      modeling.validate_ball_scores({1: 0.5}, range(1, 34), '红球')
    with self.assertRaises(ValueError):
      modeling.validate_ball_scores(
          {ball: np.nan for ball in range(1, 34)}, range(1, 34), '红球'
      )

  def test_strategy_scoring_rejects_empty_history_explicitly(self):
    history = pd.DataFrame(columns=modeling.FEATURE_COLUMNS)
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

  def test_execution_boundaries_validate_strategy_params_before_work(self):
    invalid_params = {'weight_freq': 0.9}
    with patch.object(modeling, 'validate_model_sets') as validate_models:
      with self.assertRaisesRegex(ValueError, '总和为 1'):
        modeling.run_strategy_and_get_scores(
            pd.DataFrame(columns=modeling.FEATURE_COLUMNS),
            invalid_params,
            {},
            {},
            modeling.FEATURE_COLUMNS,
        )
      validate_models.assert_not_called()

    with patch.object(backtesting, 'train_prediction_models') as train:
      with self.assertRaisesRegex(ValueError, '总和为 1'):
        backtesting.run_full_backtest(
            pd.DataFrame(),
            invalid_params,
            modeling.FEATURE_COLUMNS,
            1,
        )
      train.assert_not_called()

  def test_strategy_scores_with_single_class_training_history(self):
    history = modeling.feature_engineer(pd.DataFrame({
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
    self.assertTrue(all(np.isfinite(score) for score in red_scores.values()))
    self.assertTrue(all(np.isfinite(score) for score in blue_scores.values()))
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
    with patch.object(selection, 'score_combination', return_value=1.0) as score:
      ranked = selection.find_best_7_red_combinations(
          passed,
          range(1, 8),
          scores,
      )

    self.assertEqual(ranked, [((1, 2, 3, 4, 5, 6, 7), 1)])
    self.assertEqual(score.call_count, 1)
    self.assertEqual(score.call_args.args[0], passed[0])

  def test_backtest_result_metrics(self):
    result = backtesting.BacktestResult(
        50, 48, 480, 960, 280, {}, evaluated_periods=50,
        pool_red_hits=155, ticket_red_hit_counts=Counter({2: 360, 3: 100, 4: 20}),
        candidate_tickets=480,
        candidate_red_hit_counts=Counter({1: 100, 2: 280, 3: 80, 4: 20}),
        blue_hit_periods=4,
        rank_band_hits=Counter({"high": 30, "middle": 100, "low": 20, "other": 150}),
    )
    self.assertEqual(result.profit, -680)
    self.assertAlmostEqual(result.roi, 280 / 960)
    self.assertEqual(result.average_pool_red_hits, 3.1)
    self.assertAlmostEqual(result.average_ticket_red_hits, 2.2916666667)
    self.assertEqual(result.three_plus_red_tickets, 120)
    self.assertEqual(result.three_plus_red_rate, 0.25)
    self.assertAlmostEqual(result.average_candidate_red_hits, 980 / 480)
    self.assertAlmostEqual(result.candidate_three_plus_red_rate, 100 / 480)
    self.assertAlmostEqual(result.ranking_red_hit_delta, 0.25)
    self.assertAlmostEqual(result.ranking_three_plus_delta, 1 / 24)
    self.assertEqual(result.blue_hit_rate, 0.08)
    self.assertEqual(result.rank_band_rate("middle"), 1 / 3)
    self.assertAlmostEqual(result.rank_band_lift("middle"), 11 / 9)

  def test_backtest_rejects_invalid_controls_before_training(self):
    invalid_requests = (
        {'num_periods': -1},
        {'num_periods': True},
        {'num_periods': 1, 'pool_modes': ()},
        {'num_periods': 1, 'pool_modes': 'mixed'},
        {'num_periods': 1, 'pool_modes': ('mixed', 'mixed')},
        {'num_periods': 1, 'pool_modes': ('unknown',)},
        {'num_periods': 1, 'config': object()},
    )
    with patch.object(backtesting, 'train_prediction_models') as train:
      for request in invalid_requests:
        kwargs = {
            'full_df': pd.DataFrame(),
            'params': analyzer.DEFAULT_PARAMS,
            'feature_columns': modeling.FEATURE_COLUMNS,
            **request,
        }
        with self.subTest(request=request), self.assertRaises((TypeError, ValueError)):
          backtesting.run_full_backtest(**kwargs)
      train.assert_not_called()

  def test_legacy_backtest_api_builds_request_without_losing_options(self):
    strategy = analyzer.StrategyConfig(rejection_lib_size=17, random_seed=9)
    with patch.object(backtesting, 'run_backtest', return_value='result') as run:
      result = backtesting.run_full_backtest(
          'frame',
          {'weight': 1},
          ('feature',),
          12,
          pool_modes=('high', 'low'),
          config=strategy,
      )

    self.assertEqual(result, 'result')
    self.assertEqual(run.call_args.args[0], 'frame')
    request = run.call_args.args[1]
    self.assertIsInstance(request, backtesting.BacktestRequest)
    self.assertEqual(request.params, {'weight': 1})
    self.assertEqual(request.feature_columns, ('feature',))
    self.assertEqual(request.num_periods, 12)
    self.assertEqual(request.pool_modes, ('high', 'low'))
    self.assertIs(request.config, strategy)

  def test_backtest_result_preserves_earlier_and_recent_windows(self):
    earlier = backtesting.BacktestResult(
        2, 2, 20, 40, 0, Counter(), evaluated_periods=2, pool_red_hits=6
    )
    recent = backtesting.BacktestResult(
        3, 3, 30, 60, 0, Counter(), evaluated_periods=3, pool_red_hits=12
    )
    aggregate = backtesting.BacktestAccumulator(
        evaluated_periods=5,
        pool_red_hits=18,
    ).to_result(5, {'earlier': earlier, 'recent': recent})

    self.assertEqual(aggregate.windows['earlier'].average_pool_red_hits, 3)
    self.assertEqual(aggregate.windows['recent'].average_pool_red_hits, 4)

  def test_backtest_result_is_a_snapshot_of_accumulated_metrics(self):
    accumulator = backtesting.BacktestAccumulator(
        prize_counts=Counter({(3, 1): 1}),
        ticket_red_hit_counts=Counter({3: 2}),
        candidate_red_hit_counts=Counter({2: 4}),
        rank_band_hits=Counter({'middle': 3}),
    )
    windows = {'earlier': backtesting.BacktestResult(
        1, 0, 0, 0, 0, Counter()
    )}
    result = accumulator.to_result(1, windows)

    accumulator.prize_counts[(3, 1)] += 1
    accumulator.ticket_red_hit_counts[3] += 1
    accumulator.candidate_red_hit_counts[2] += 1
    accumulator.rank_band_hits['middle'] += 1
    accumulator.rank_band_widths['middle'] = 99
    windows.clear()

    self.assertEqual(result.prize_counts, {(3, 1): 1})
    self.assertEqual(result.ticket_red_hit_counts, {3: 2})
    self.assertEqual(result.candidate_red_hit_counts, {2: 4})
    self.assertEqual(result.rank_band_hits, {'middle': 3})
    self.assertEqual(result.rank_band_widths['middle'], 9)
    self.assertIn('earlier', result.windows)

  def test_backtest_selection_reuses_combo_hits_across_accumulators(self):
    selection_result = SimpleNamespace(
        red_pool=(1, 2, 3, 4, 5, 6, 7),
        passed_combos=((1, 2, 3, 4, 5, 6),),
        recommendations=((1, 2, 3, 4, 5, 6),),
    )
    issue = backtesting.BacktestIssue(
        actual_reds=frozenset((1, 2, 3, 4, 5, 6)),
        actual_blue=8,
        recommended_blue=8,
        rank_band_hits=Counter({'middle': 6}),
    )
    total = backtesting.BacktestAccumulator()
    window = backtesting.BacktestAccumulator()

    hits = backtesting.record_backtest_selection(total, selection_result, issue)
    reused_hits = backtesting.record_backtest_selection(
        window,
        selection_result,
        issue,
        hits,
    )

    self.assertIs(reused_hits, hits)
    self.assertEqual(total.to_result(1), window.to_result(1))
    self.assertEqual(total.pool_red_hits, 6)
    self.assertEqual(total.blue_hit_periods, 1)

  def test_full_backtest_aggregate_equals_sum_of_stability_windows(self):
    frame = pd.DataFrame({
        '期号': list(range(2025001, 2025053)),
        '红球': [[7, 8, 9, 10, 11, 12] for _ in range(50)] + [
            [1, 2, 3, 8, 9, 10],
            [1, 2, 3, 4, 5, 6],
        ],
        '蓝球': [1 for _ in range(50)] + [7, 8],
    })
    red_scores = {ball: float(34 - ball) for ball in range(1, 34)}
    blue_scores = {ball: float(ball == 8) for ball in range(1, 17)}
    models = (
        {ball: object() for ball in range(1, 34)},
        {ball: object() for ball in range(1, 17)},
    )
    strategy = analyzer.StrategyConfig(
        pool_size_red=6,
        high_count=2,
        low_count=2,
        recommendation_count=1,
        rejection_lib_size=0,
    )

    with (
        patch.object(backtesting, 'train_prediction_models', return_value=models),
        patch.object(
            backtesting,
            'run_strategy_and_get_scores',
            return_value=(red_scores, blue_scores),
        ),
        patch.object(auditing, 'get_omission', return_value={}),
        patch.object(backtesting, 'make_rejection_set', return_value=set()),
        patch.object(
            selection, 'build_red_pool', return_value=[1, 2, 3, 4, 5, 6]
        ),
        patch.object(selection, 'passes_red_filters', return_value=True),
    ):
      total = backtesting.run_full_backtest(
          frame,
          analyzer.DEFAULT_PARAMS,
          [],
          2,
          config=strategy,
      )['mixed']

    windows = (total.windows['earlier'], total.windows['recent'])
    scalar_fields = (
        'periods', 'active_periods', 'evaluated_periods', 'tickets', 'cost',
        'winnings', 'pool_red_hits', 'candidate_tickets', 'blue_hit_periods',
    )
    counter_fields = (
        'prize_counts', 'ticket_red_hit_counts', 'candidate_red_hit_counts',
        'rank_band_hits',
    )

    for field_name in scalar_fields:
      self.assertEqual(
          getattr(total, field_name),
          sum(getattr(window, field_name) for window in windows),
          field_name,
      )
    for field_name in counter_fields:
      self.assertEqual(
          getattr(total, field_name),
          sum(
              (getattr(window, field_name) for window in windows),
              Counter(),
          ),
          field_name,
      )

  def test_backtest_skips_equal_length_model_sets_with_wrong_ball_keys(self):
    frame = pd.DataFrame({
        '期号': list(range(2025001, 2025052)),
        '红球': [[1, 2, 3, 4, 5, 6] for _ in range(51)],
        '蓝球': [1 for _ in range(51)],
    })
    red_models = {ball: object() for ball in range(1, 33)}
    red_models[34] = object()
    blue_models = {ball: object() for ball in range(1, 17)}

    with (
        patch.object(
            backtesting,
            'train_prediction_models',
            return_value=(red_models, blue_models),
        ),
        patch.object(backtesting, 'run_strategy_and_get_scores') as score,
    ):
      result = backtesting.run_full_backtest(
          frame,
          analyzer.DEFAULT_PARAMS,
          modeling.FEATURE_COLUMNS,
          1,
          config=analyzer.StrategyConfig(rejection_lib_size=0),
      )['mixed']

    self.assertEqual(result.periods, 1)
    self.assertEqual(result.evaluated_periods, 0)
    score.assert_not_called()

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

  def test_backtest_tracks_evaluation_and_hit_quality(self):
    frame = pd.DataFrame({
        "期号": list(range(2025001, 2025052)),
        "红球": [[7, 8, 9, 10, 11, 12] for _ in range(50)]
                + [[1, 13, 22, 30, 32, 33]],
        "蓝球": [1 for _ in range(50)] + [7],
    })
    red_scores = {ball: float(34 - ball) for ball in range(1, 34)}
    blue_scores = {ball: float(ball == 7) for ball in range(1, 17)}
    red_models = {ball: object() for ball in range(1, 34)}
    blue_models = {ball: object() for ball in range(1, 17)}
    config = analyzer.StrategyConfig(
        pool_size_red=6,
        high_count=2,
        low_count=2,
        recommendation_count=3,
        rejection_lib_size=1234,
        random_seed=99,
    )

    with (
        patch.object(
            backtesting,
            "train_prediction_models",
            return_value=(red_models, blue_models),
        ) as train_mock,
        patch.object(
            backtesting,
            "run_strategy_and_get_scores",
            return_value=(red_scores, blue_scores),
        ),
        patch.object(backtesting, "rejection_seed_for_issue", return_value=123) as seed_mock,
        patch.object(backtesting, "make_rejection_set", return_value=set()) as rejection_mock,
        patch.object(auditing, "get_omission", return_value={}),
        patch.object(
            selection, "build_red_pool", return_value=[1, 2, 3, 4, 13, 14]
        ) as pool_mock,
        patch.object(selection, "passes_red_filters", return_value=True),
        patch.object(
            selection,
            "select_recommendation_portfolio",
            return_value=[(1, 2, 3, 4, 13, 14)],
        ) as selection_mock,
    ):
      result = backtesting.run_full_backtest(
          frame, analyzer.DEFAULT_PARAMS, [], 1, config=config
      )["mixed"]

    seed_mock.assert_called_once_with(99, 2025051)
    training_frame = train_mock.call_args.args[0]
    self.assertEqual(training_frame.iloc[-1]['期号'], 2025050)
    self.assertNotIn(2025051, training_frame['期号'].tolist())
    self.assertEqual(rejection_mock.call_args.args[0], 1234)
    pool_mock.assert_called_once_with(red_scores, config=config, mode="mixed")
    self.assertEqual(selection_mock.call_args.args[0].limit, 3)
    self.assertEqual(result.periods, 1)
    self.assertEqual(result.evaluated_periods, 1)
    self.assertEqual(result.active_periods, 1)
    self.assertEqual(result.pool_red_hits, 2)
    self.assertEqual(result.ticket_red_hit_counts, {2: 1})
    self.assertEqual(result.candidate_tickets, 1)
    self.assertEqual(result.candidate_red_hit_counts, {2: 1})
    self.assertEqual(result.ranking_red_hit_delta, 0)
    self.assertEqual(result.blue_hit_periods, 1)
    self.assertEqual(
        result.rank_band_hits,
        {"high": 1, "middle": 0, "low": 2, "other": 3},
    )
    self.assertEqual(
        result.rank_band_widths,
        {'high': 2, 'middle': 2, 'low': 2, 'other': 27},
    )

  def test_red_score_adjustments_restore_original_bonuses(self):
    history = pd.DataFrame({'红球': [
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

  def test_historical_rule_audit_reports_requested_window(self):
    frame = pd.DataFrame({
        '红球': [[1, 5, 10, 18, 25, 31] for _ in range(12)],
        '蓝球': [1 for _ in range(12)],
    })
    result = backtesting.audit_historical_rule_coverage(frame, periods=2)
    self.assertEqual(set(result), set(analyzer.FILTER_NAMES))
    self.assertTrue(all(item['total'] == 2 for item in result.values()))

  def test_historical_hard_pipeline_reports_incremental_losses(self):
    frame = pd.DataFrame({
        '红球': [[3, 4, 5, 6, 7, 8] for _ in range(10)] + [
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8],
        ],
    })
    custom_rules = (
        rules.RuleDefinition('first', True, lambda combo, *_: combo[0] >= 2),
        rules.RuleDefinition('second', True, lambda combo, *_: combo[-1] % 2 == 0),
        rules.RuleDefinition('soft', False, lambda combo, *_: False),
    )
    with patch.object(auditing, 'RED_RULES', custom_rules):
      result = backtesting.audit_historical_hard_pipeline(frame, periods=3)

    self.assertEqual(result['total'], 3)
    self.assertEqual(result['passed'], 1)
    self.assertAlmostEqual(result['rate'], 1 / 3)
    self.assertEqual(result['stages'], [
        {'rule': 'first', 'before': 3, 'removed': 1, 'remaining': 2},
        {'rule': 'second', 'before': 2, 'removed': 1, 'remaining': 1},
    ])


if __name__ == "__main__":
    unittest.main()
