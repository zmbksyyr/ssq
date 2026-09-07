import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ssq_data_sources as sources
import ssq_source_comparison as comparison


class SourceComparisonTests(unittest.TestCase):
    def test_data_sources_preserves_comparison_compatibility_exports(self):
        self.assertIs(sources.cross_check_sources, comparison.cross_check_sources)
        self.assertIs(
            sources.find_secondary_only_issues,
            comparison.find_secondary_only_issues,
        )

    def test_cross_check_reports_only_overlapping_number_mismatches(self):
        primary = [
            {'期号': '2026001', '红球': '01,02,03,04,05,06', '蓝球': '07'},
            {'期号': '2026002', '红球': '02,03,04,05,06,07', '蓝球': '08'},
        ]
        secondary = [
            {'期号': '2026001', '红球': '01,02,03,04,05,06', '蓝球': '09'},
            {'期号': '2026003', '红球': '03,04,05,06,07,08', '蓝球': '10'},
        ]

        self.assertEqual(
            comparison.cross_check_sources(primary, secondary),
            ['2026001'],
        )

    def test_secondary_only_issues_are_deduplicated_and_sorted_numerically(self):
        primary = [{'期号': '2026001'}]
        secondary = [
            {'期号': '2026010'},
            {'期号': '2026002'},
            {'期号': '2026010'},
            {'期号': '2026001'},
        ]

        self.assertEqual(
            comparison.find_secondary_only_issues(primary, secondary),
            ['2026002', '2026010'],
        )


if __name__ == '__main__':
    unittest.main()
