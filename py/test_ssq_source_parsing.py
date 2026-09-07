import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ssq_data_sources as sources
import ssq_source_parsing as source_parsing


class SourceParsingTests(unittest.TestCase):
    def test_data_sources_preserves_parser_compatibility_exports(self):
        self.assertIs(sources.parse_html_data, source_parsing.parse_html_data)
        self.assertIs(sources.parse_txt_data, source_parsing.parse_txt_data)

    def test_html_parser_normalizes_valid_rows_and_skips_invalid_rows(self):
        content = (
            '<table><tr><th>期号</th><th>红球</th><th>蓝球</th></tr>'
            '<tr><td>2026001期</td><td>6 1 10 3 8 2</td><td>9</td></tr>'
            '<tr><td>999期</td><td>1 2 3 4 5 6</td><td>7</td></tr>'
            '</table>'
        )

        self.assertEqual(source_parsing.parse_html_data(content), [{
            '期号': '2026001',
            '红球': '01,02,03,06,08,10',
            '蓝球': '09',
        }])

    def test_html_parser_rejects_content_without_a_table(self):
        self.assertEqual(source_parsing.parse_html_data('<html></html>'), [])

    def test_txt_parser_requires_canonical_date_and_valid_balls(self):
        rows = source_parsing.parse_txt_data([
            '2026001 2026-01-01 6 1 10 3 8 2 9',
            '2026002 2026-1-2 1 2 3 4 5 6 7',
            '2026003 2026-01-04 1 2 3 4 5 34 7',
        ])

        self.assertEqual(rows, [[
            '2026001', '2026-01-01', '01,02,03,06,08,10', '09',
        ]])


if __name__ == '__main__':
    unittest.main()
