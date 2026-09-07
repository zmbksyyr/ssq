import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ssq_core import (
    atomic_write_text, infer_next_issue, parse_blue_ball, parse_blue_balls,
    parse_issue, parse_red_balls,
)


class CoreValidationTests(unittest.TestCase):
    def test_parses_and_sorts_valid_red_balls(self):
        self.assertEqual(parse_red_balls('06,01,10,03,08,02'), [1, 2, 3, 6, 8, 10])

    def test_rejects_invalid_red_balls(self):
        for value in ('1,2,3,4,5', '1,1,2,3,4,5', '1,2,3,4,5,34', [1, 2, 3, 4, 5, 6.5]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_red_balls(value)

    def test_rejects_invalid_blue_ball(self):
        self.assertEqual(parse_blue_ball('09'), 9)
        for value in (0, 17, 9.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_blue_ball(value)

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


if __name__ == '__main__':
    unittest.main()
