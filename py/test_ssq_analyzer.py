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
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ssq_analyzer as analyzer
import ssq_backtesting as backtesting
import ssq_config as config
import ssq_modeling as modeling
import ssq_rules as rules
import ssq_selection as selection
import ssq_workflow as workflow


class AnalyzerTests(unittest.TestCase):
  def test_cli_options_build_runtime_configuration(self):
    defaults = config.parse_cli_options([])
    self.assertEqual(defaults.rejection_size, 500_000)
    self.assertEqual(defaults.backtest_pool_modes, ('mixed',))
    self.assertEqual(defaults.strategy_config.rejection_lib_size, 500_000)

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

  def test_analyzer_reexports_modeling_functions(self):
    self.assertIs(analyzer.feature_engineer, modeling.feature_engineer)
    self.assertIs(
        analyzer.run_strategy_and_get_scores,
        modeling.run_strategy_and_get_scores,
    )

  def test_analyzer_reexports_selection_functions(self):
    self.assertIs(analyzer.build_red_pool, selection.build_red_pool)
    self.assertIs(analyzer.generate_red_candidates, selection.generate_red_candidates)

  def test_analyzer_reexports_backtesting_functions(self):
    self.assertIs(analyzer.BacktestResult, backtesting.BacktestResult)
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
        pipeline_stats=[],
        rule_coverage=coverage,
        hard_pipeline_coverage={
            'stages': [], 'passed': 0, 'total': 0, 'rate': 0.0,
        },
        rule_audit_periods=200,
        selection=candidate_selection,
        recommended_blues=[],
        best_7_reds=[],
    )

    report = analyzer.build_analysis_report(data)

    self.assertIn('Data_Basis_Issue: 2026103', report)
    self.assertIn('报告生成时间: 2026-09-07 12:34:56', report)
    self.assertIn('模式: 使用内置的默认参数', report)
    self.assertIn('[软] prime_composite_ratio', report)
    self.assertIn('未能生成足够的单式组合', report)
    self.assertIn('未能生成足够的复式组合', report)

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
    self.assertIn('red_sum', result.columns)

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

  def test_candidate_generation_uses_one_selection_pipeline(self):
    scores = {ball: float(ball) for ball in range(1, 34)}
    context = rules.RuleContext(last_draw={1}, previous_draw={2})
    rejected = {(1, 2, 3, 4, 5, 6)}
    config = analyzer.StrategyConfig(
        pool_size_red=7,
        high_count=2,
        low_count=2,
        recommendation_count=2,
        rejection_lib_size=1,
    )

    def fake_filter(combo, actual_context, actual_rejection):
      self.assertIs(actual_context, context)
      self.assertIs(actual_rejection, rejected)
      return combo not in actual_rejection

    def fake_select(combos, *_args, **_kwargs):
      return list(combos)[:2]

    with (
        patch.object(selection, 'build_red_pool', return_value=list(range(1, 8))),
        patch.object(selection, 'passes_red_filters', side_effect=fake_filter) as check,
        patch.object(selection, 'select_recommendations', side_effect=fake_select) as select,
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
    self.assertEqual(select.call_args.kwargs['limit'], 2)

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
      analyzer.StrategyConfig(rejection_lib_size=-1)


  def test_rejection_set_is_reproducible(self):
    first = selection.make_rejection_set(100, random.Random(7))
    second = selection.make_rejection_set(100, random.Random(7))
    self.assertEqual(first, second)
    self.assertEqual(len(first), 100)
    with self.assertRaises(ValueError):
      selection.make_rejection_set(config.TOTAL_RED_COMBINATIONS + 1)

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

  def test_positive_probability_uses_the_positive_class_column(self):
    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    features = pd.DataFrame({'value': range(20)})
    model.fit(features, [0, 1] * 10)
    expected = model.predict_proba(features.iloc[:2])[:, 1]
    actual = modeling.predict_positive_probability(model, features.iloc[:2])
    self.assertTrue(np.array_equal(actual, expected))

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

  def test_duplex_selection_uses_score_to_break_coverage_ties(self):
    pool = list(range(1, 9))
    passed = list(combinations(pool, 6))
    scores = {n: float(34 - n) for n in range(1, 34)}
    ranked = selection.find_best_7_red_combinations(passed, pool, scores)
    self.assertEqual(ranked[0], ((2, 3, 4, 5, 6, 7, 8), 7))

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
        patch.object(backtesting, "get_omission", return_value={}),
        patch.object(
            selection, "build_red_pool", return_value=[1, 2, 3, 4, 13, 14]
        ) as pool_mock,
        patch.object(selection, "passes_red_filters", return_value=True),
        patch.object(
            selection,
            "select_recommendations",
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
    self.assertEqual(selection_mock.call_args.kwargs['limit'], 3)
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
        {"high": 1, "middle": 1, "low": 3, "other": 1},
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
    with patch.object(backtesting, 'RED_RULES', custom_rules):
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
