import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ssq_bonus_calculation as bonus


class BonusCalculationTests(unittest.TestCase):
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
            latest = bonus.load_latest_draw(path)
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
                bonus.load_latest_draw(path)
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
        singles, duplex = bonus.parse_report_bets(path)
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
                bonus.parse_report_bets(path)
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
                bonus.parse_report_bets(path)
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
                bonus.parse_report_bets(path)
        finally:
            Path(path).unlink()

    def test_duplex_counts_hit_and_missed_blue_subtickets(self):
        total, breakdown, _ = bonus.calculate_duplex_prize(
            [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 8],
            {1, 2, 3, 4, 5, 6}, 8
        )
        self.assertEqual(total, 5_625_200)
        self.assertEqual(sum(breakdown.values()), 49)

    def test_duplex_multiplies_nonwinning_blue_options(self):
        total, breakdown, _ = bonus.calculate_duplex_prize(
            [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7],
            {1, 2, 3, 4, 5, 6}, 8
        )
        self.assertEqual(total, 708_400)
        self.assertEqual(sum(breakdown.values()), 49)

    def test_prize_calculators_reject_invalid_direct_inputs(self):
        with self.assertRaisesRegex(ValueError, '不能重复'):
            bonus.calculate_single_prize(
                [1, 1, 2, 3, 4, 5], 7, {1, 2, 3, 4, 5, 6}, 7
            )
        with self.assertRaisesRegex(ValueError, '不能重复'):
            bonus.calculate_duplex_prize(
                [1, 2, 3, 4, 5, 6, 7], [1, 1],
                {1, 2, 3, 4, 5, 6}, 7,
            )


if __name__ == '__main__':
    unittest.main()
