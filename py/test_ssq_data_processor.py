import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ssq_data_processor as processor


class DataProcessorTests(unittest.TestCase):
    def test_parse_txt_data_validates_and_normalizes(self):
        rows = processor.parse_txt_data([
            '2026001 2026-01-01 6 1 10 3 8 2 9',
            '2026002 2026-01-03 1 1 2 3 4 5 6',
            '2026003 invalid 1 2 3 4 5 6 7',
            '2026004 2026-01-05 1 2 3 4 5 34 7',
        ])
        self.assertEqual(rows, [[
            '2026001', '2026-01-01', '01,02,03,06,08,10', '09'
        ]])


if __name__ == '__main__':
    unittest.main()
