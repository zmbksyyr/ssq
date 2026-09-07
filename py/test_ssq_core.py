import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ssq_core import parse_blue_ball, parse_blue_balls, parse_red_balls


class CoreValidationTests(unittest.TestCase):
    def test_parses_and_sorts_valid_red_balls(self):
        self.assertEqual(parse_red_balls('06,01,10,03,08,02'), [1, 2, 3, 6, 8, 10])

    def test_rejects_invalid_red_balls(self):
        for value in ('1,2,3,4,5', '1,1,2,3,4,5', '1,2,3,4,5,34'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_red_balls(value)

    def test_rejects_invalid_blue_ball(self):
        self.assertEqual(parse_blue_ball('09'), 9)
        for value in (0, 17):
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


if __name__ == '__main__':
    unittest.main()
