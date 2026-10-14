"""Consolidated bonus_check tests (merged from 4 module test files).

Originally one test module per production module. The production modules are now
merged into py/ssq_bonus_check.py, so the tests are grouped the same way. Retired module
names still resolve through ssq_compat, which run_tests.py sets up before these
modules are imported.

Run with: python py/run_tests.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ast
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas
import ssq_bet_parsing as bet_parsing
import ssq_bonus_workflow as bonus_workflow
import ssq_latest_draw as latest_draw
import ssq_prize_calculation as prize_calculation
import ssq_prizes as prizes
import ssq_report_discovery as report_discovery
from ssq_domain import PRIZE_NAMES, PRIZE_RULES

# ===== from test_ssq_bet_parsing.py =====

VALID_REPORT = """【单式推荐 (2组)】
组合 1: 红球 [1, 2, 3, 4, 5, 6] 蓝球 [07]
组合 2: 红球 [7, 8, 9, 10, 11, 12] 蓝球 [16]
【7+N 复式推荐 (1组)】
红球: [1, 2, 3, 4, 5, 6, 7]
蓝球: [1, 3, 5, 7, 9, 11, 13]
"""


class BetParsingTests(unittest.TestCase):
    def test_prizes_preserves_parser_compatibility_exports(self):
        for name in (
            'SINGLE_HEADER_PATTERN',
            'DUPLEX_HEADER_PATTERN',
            'SINGLE_BET_PATTERN',
            'parse_single_bet_line',
            'parse_duplex_section',
            'validate_parsed_bets',
            'parse_report_bets',
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(prizes, name), getattr(bet_parsing, name))

    def test_report_content_parser_returns_all_validated_bets(self):
        singles, duplex = bet_parsing.parse_report_bets_content(VALID_REPORT)

        self.assertEqual(len(singles), 2)
        self.assertEqual(singles[0], {'red': [1, 2, 3, 4, 5, 6], 'blue': 7})
        self.assertEqual(duplex['red'], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(duplex['blue'], [1, 3, 5, 7, 9, 11, 13])

    def test_file_parser_delegates_to_content_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'analysis.txt'
            path.write_text(VALID_REPORT, encoding='utf-8')

            self.assertEqual(
                bet_parsing.parse_report_bets(path),
                bet_parsing.parse_report_bets_content(VALID_REPORT),
            )

    def test_report_rejects_duplicate_single_bets(self):
        duplicate = VALID_REPORT.replace(
            '组合 2: 红球 [7, 8, 9, 10, 11, 12] 蓝球 [16]',
            '组合 2: 红球 [1, 2, 3, 4, 5, 6] 蓝球 [07]',
        )

        with self.assertRaisesRegex(ValueError, '重复单式投注'):
            bet_parsing.parse_report_bets_content(duplicate)


# ===== from test_ssq_latest_draw.py =====


class LatestDrawTests(unittest.TestCase):
    def test_workflow_wrapper_delegates_its_validation_dependencies(self):
        with patch.object(
            latest_draw,
            'load_latest_draw',
            return_value={'issue': 2026001},
        ) as load:
            result = bonus_workflow.load_latest_draw('draws.csv')

        self.assertEqual(result, {'issue': 2026001})
        load.assert_called_once_with(
            'draws.csv',
            normalize=bonus_workflow.normalize_draw_frame,
            validate_dates=bonus_workflow.validate_draw_dates_not_future,
        )

    def test_loader_normalizes_and_validates_before_selecting_latest(self):
        normalized = pandas.DataFrame([{
            '期号': 2026002,
            '日期': '2026-01-04',
            '红球': [7, 8, 9, 10, 11, 12],
            '蓝球': 16,
        }])
        normalize = Mock(return_value=normalized)
        validate_dates = Mock()

        with patch.object(pandas, 'read_csv', return_value=pandas.DataFrame()) as read:
            result = latest_draw.load_latest_draw(
                'draws.csv',
                normalize=normalize,
                validate_dates=validate_dates,
            )

        read.assert_called_once_with('draws.csv', header=0)
        normalize.assert_called_once_with(read.return_value)
        validate_dates.assert_called_once_with(normalized)
        self.assertEqual(result, {
            'issue': 2026002,
            'red': {7, 8, 9, 10, 11, 12},
            'blue': 16,
        })

    def test_loader_rejects_empty_normalized_history(self):
        with (
            patch.object(pandas, 'read_csv', return_value=pandas.DataFrame()),
            self.assertRaisesRegex(ValueError, '开奖数据为空'),
        ):
            latest_draw.load_latest_draw(
                'draws.csv',
                normalize=Mock(return_value=pandas.DataFrame()),
                validate_dates=Mock(),
            )


# ===== from test_ssq_prize_calculation.py =====


class PrizeCalculationTests(unittest.TestCase):
    def test_runtime_modules_do_not_depend_on_prizes_compatibility_facade(self):
        module_dir = Path(__file__).parent
        offenders = []
        for path in module_dir.glob('ssq_*.py'):
            if path.name == 'ssq_prizes.py':
                continue
            tree = ast.parse(path.read_text(encoding='utf-8'))
            if any(
                isinstance(node, (ast.Import, ast.ImportFrom))
                and (
                    getattr(node, 'module', None) == 'ssq_prizes'
                    or any(alias.name == 'ssq_prizes' for alias in node.names)
                )
                for node in ast.walk(tree)
            ):
                offenders.append(path.name)

        self.assertEqual(offenders, [])

    def test_prizes_preserves_calculator_compatibility_exports(self):
        self.assertIs(
            prizes.calculate_single_prize,
            prize_calculation.calculate_single_prize,
        )
        self.assertIs(
            prizes.calculate_duplex_prize,
            prize_calculation.calculate_duplex_prize,
        )

    def test_single_ticket_covers_every_prize_rule_and_no_prize(self):
        winning_reds = [1, 2, 3, 4, 5, 6]
        losing_reds = [7, 8, 9, 10, 11, 12]
        cases = [*PRIZE_RULES.items(), ((3, 0), 0)]

        for (red_hits, blue_hit), expected_prize in cases:
            with self.subTest(red_hits=red_hits, blue_hit=blue_hit):
                bet_reds = winning_reds[:red_hits] + losing_reds[:6 - red_hits]
                prize, name, summary = prize_calculation.calculate_single_prize(
                    bet_reds,
                    7 if blue_hit else 8,
                    winning_reds,
                    7,
                )

                self.assertEqual(prize, expected_prize)
                self.assertEqual(
                    name,
                    PRIZE_NAMES.get((red_hits, blue_hit), '未中奖'),
                )
                self.assertEqual(summary, f'命中{red_hits}+{blue_hit}')

    def test_duplex_counts_every_contained_standard_ticket(self):
        total, breakdown, summary = prize_calculation.calculate_duplex_prize(
            [1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3],
            [1, 2, 3, 4, 5, 6],
            1,
        )

        self.assertEqual(sum(breakdown.values()), 21)
        self.assertEqual(total, 5_220_400)
        self.assertEqual(summary, '总计命中 6 个红球, 1 个蓝球')


# ===== from test_ssq_report_discovery.py =====


class ReportDiscoveryTests(unittest.TestCase):
    def test_workflow_preserves_metadata_parser_export(self):
        self.assertIs(
            bonus_workflow.parse_report_target_issue,
            report_discovery.parse_report_target_issue,
        )

    def test_target_issue_metadata_must_be_unique_and_valid(self):
        self.assertEqual(
            report_discovery.parse_report_target_issue(
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
                report_discovery.parse_report_target_issue(content)

    def test_discovery_skips_newer_invalid_and_wrong_issue_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory)
            matching = report_dir / 'ssq_analysis_output_20260101_000000.txt'
            wrong = report_dir / 'ssq_analysis_output_20260102_000000.txt'
            invalid = report_dir / 'ssq_analysis_output_20260103_000000.txt'
            matching.write_text(
                'Prediction_Target_Issue: 2026001\n', encoding='utf-8'
            )
            wrong.write_text(
                'Prediction_Target_Issue: 2026002\n', encoding='utf-8'
            )
            invalid.write_text(
                'Prediction_Target_Issue: 2026001\n'
                'Prediction_Target_Issue: 2026002\n',
                encoding='utf-8',
            )

            result, error = report_discovery.find_matching_report(2026001, report_dir)

        self.assertIsNone(error)
        self.assertEqual(Path(result), matching)

    def test_discovery_distinguishes_empty_directory_from_no_match(self):
        with tempfile.TemporaryDirectory() as directory:
            result, error = report_discovery.find_matching_report(2026001, directory)
            self.assertIsNone(result)
            self.assertIn('未找到任何', error)

            path = Path(directory) / 'ssq_analysis_output_20260101_000000.txt'
            path.write_text(
                'Prediction_Target_Issue: 2026002\n', encoding='utf-8'
            )
            result, error = report_discovery.find_matching_report(2026001, directory)

        self.assertIsNone(result)
        self.assertIn('未找到预测目标期号', error)


if __name__ == "__main__":
    unittest.main()
