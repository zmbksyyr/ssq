import random
import sys
import unittest
import tempfile
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ssq_analyzer as analyzer


class AnalyzerTests(unittest.TestCase):
  def test_loader_rejects_issue_date_year_mismatch(self):
    content = (
      '期号,日期,红球,蓝球\n'
      '2026001,2025-12-31,"01,02,03,04,05,06",07\n'
    )
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as handle:
      handle.write(content)
      path = handle.name
    try:
      self.assertIsNone(analyzer.load_and_preprocess_data(path))
    finally:
      Path(path).unlink()

  def test_prime_definition(self):
    self.assertFalse(analyzer.is_prime(1))
    self.assertTrue(analyzer.is_prime(2))
    self.assertTrue(analyzer.is_prime(31))


  def test_build_red_pool_mixes_score_bands(self):
    scores = {ball: float(34 - ball) for ball in range(1, 34)}
    pool = analyzer.build_red_pool(scores)
    self.assertEqual(len(pool), 17)
    self.assertTrue(set(range(1, 5)).issubset(pool))
    self.assertTrue(set(range(30, 34)).issubset(pool))
    self.assertTrue(set(range(13, 22)).issubset(pool))

  def test_build_red_pool_modes(self):
    scores = {ball: float(34 - ball) for ball in range(1, 34)}
    self.assertEqual(analyzer.build_red_pool(scores, mode="high"), list(range(1, 18)))
    self.assertEqual(analyzer.build_red_pool(scores, mode="middle"), list(range(9, 26)))
    self.assertEqual(analyzer.build_red_pool(scores, mode="low"), list(range(17, 34)))
    with self.assertRaises(ValueError):
      analyzer.build_red_pool(scores, mode="invalid")


  def test_rejection_set_is_reproducible(self):
    first = analyzer.make_rejection_set(100, random.Random(7))
    second = analyzer.make_rejection_set(100, random.Random(7))
    self.assertEqual(first, second)
    self.assertEqual(len(first), 100)
    with self.assertRaises(ValueError):
      analyzer.make_rejection_set(analyzer.TOTAL_RED_COMBINATIONS + 1)


  def test_filter_explanation_includes_prime_rule(self):
    combo = (1, 2, 4, 6, 8, 10)
    failures = analyzer.explain_filter_failures(
        combo, {n: 0 for n in range(1, 34)}, [], set(), set(), set()
    )
    self.assertIn("prime_composite_ratio", failures)

  def test_all_rules_use_one_unique_registry(self):
    names = [rule.name for rule in analyzer.RED_RULES]
    self.assertEqual(len(names), len(set(names)))
    self.assertEqual(tuple(names), analyzer.FILTER_NAMES)
    prime_rule = next(rule for rule in analyzer.RED_RULES if rule.name == 'prime_composite_ratio')
    self.assertFalse(prime_rule.hard)

  def test_soft_rule_failure_does_not_reject_combination(self):
    combo = (1, 4, 8, 16, 25, 30)
    self.assertFalse(analyzer.filter_prime_composite_ratio(combo))
    self.assertTrue(analyzer.passes_red_filters(
        combo, {n: 0 for n in range(1, 34)}, [], {1}, set(), set()
    ))

  def test_filter_pipeline_stats_are_incremental(self):
    stats = analyzer.filter_pipeline_stats(
        [(3, 8, 13, 20, 27, 31), (1, 2, 3, 4, 5, 6)],
        {n: 0 for n in range(1, 34)}, [], set(), set(), set()
    )
    self.assertGreater(len(stats), 10)
    for item in stats:
      self.assertGreaterEqual(item["before"], item["remaining"])

  def test_combination_score_is_deterministic(self):
    scores = {n: n / 33 for n in range(1, 34)}
    combo = (3, 8, 14, 21, 27, 32)
    first = analyzer.score_red_combination(combo, scores)
    second = analyzer.score_red_combination(combo, scores)
    self.assertEqual(first, second)

  def test_rank_signal_prefers_center_over_both_extremes(self):
    scores = {n: float(34 - n) for n in range(1, 34)}
    middle = analyzer.score_rank_center_preference((14, 15, 16, 17, 18, 19), scores)
    high = analyzer.score_rank_center_preference((1, 2, 3, 4, 5, 6), scores)
    low = analyzer.score_rank_center_preference((28, 29, 30, 31, 32, 33), scores)
    self.assertGreater(middle, high)
    self.assertGreater(middle, low)
    cached = analyzer.build_rank_center_scores(scores)
    self.assertEqual(
      middle,
      analyzer.score_rank_center_preference((14, 15, 16, 17, 18, 19), scores, cached),
    )

  def test_recommendation_portfolio_limits_overlap(self):
    combos = list(analyzer.combinations(range(1, 13), 6))
    scores = {n: n / 12 for n in range(1, 13)}
    selected = analyzer.select_recommendations(combos, scores, limit=10, max_shared=4)
    self.assertEqual(len(selected), 10)
    for index, combo in enumerate(selected):
      for other in selected[index + 1:]:
        self.assertLessEqual(len(set(combo) & set(other)), 4)

  def test_duplex_selection_uses_score_to_break_coverage_ties(self):
    pool = list(range(1, 9))
    passed = list(analyzer.combinations(pool, 6))
    scores = {n: float(34 - n) for n in range(1, 34)}
    ranked = analyzer.find_best_7_red_combinations(passed, pool, scores)
    self.assertEqual(ranked[0], ((2, 3, 4, 5, 6, 7, 8), 7))

  def test_backtest_result_metrics(self):
    result = analyzer.BacktestResult(50, 50, 500, 1000, 280, {})
    self.assertEqual(result.profit, -720)
    self.assertEqual(result.roi, 0.28)

  def test_red_score_adjustments_restore_original_bonuses(self):
    history = pd.DataFrame({'红球': [
        [1, 2, 3, 4, 5, 6],
        [1, 7, 8, 9, 10, 11],
    ]})
    params = {
        'hot_lookback': 2, 'hot_threshold': 2, 'hot_bonus': 2.0,
        'cold_lookback': 2, 'cold_bonus': 3.0, 'repeat_bonus': 5.0,
    }
    adjusted = analyzer.apply_red_score_adjustments(
        {ball: 1.0 for ball in range(1, 34)}, history, params
    )
    self.assertEqual(adjusted[1], 10.0)  # hot and repeated
    self.assertEqual(adjusted[7], 5.0)   # repeated only
    self.assertEqual(adjusted[12], 3.0)  # cold only
    self.assertEqual(adjusted[2], 1.0)   # no adjustment

  def test_strategy_params_are_validated_and_completed(self):
    params = analyzer.validate_strategy_params({})
    self.assertEqual(params['weight_freq'], 0.3)
    with self.assertRaises(ValueError):
      analyzer.validate_strategy_params({'weight_freq': 0.9})
    with self.assertRaises(ValueError):
      analyzer.validate_strategy_params({'decay_factor': 1.1})
    with self.assertRaises(ValueError):
      analyzer.validate_strategy_params({'repeat_bonus': 0})

  def test_historical_rule_audit_reports_requested_window(self):
    frame = pd.DataFrame({
        '红球': [[1, 5, 10, 18, 25, 31] for _ in range(12)],
        '蓝球': [1 for _ in range(12)],
    })
    result = analyzer.audit_historical_rule_coverage(frame, periods=2)
    self.assertEqual(set(result), set(analyzer.FILTER_NAMES))
    self.assertTrue(all(item['total'] == 2 for item in result.values()))


if __name__ == "__main__":
    unittest.main()
