import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ssq_bet_parsing as bet_parsing
import ssq_prizes as prizes

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


if __name__ == '__main__':
    unittest.main()
