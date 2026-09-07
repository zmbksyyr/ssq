import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ssq_csv_store as csv_store
import ssq_data_store


class CsvStoreTests(unittest.TestCase):
    def test_data_store_preserves_csv_compatibility_exports(self):
        self.assertIs(ssq_data_store.read_existing_csv, csv_store.read_existing_csv)
        self.assertIs(ssq_data_store.atomic_write_csv, csv_store.atomic_write_csv)

    def test_missing_and_empty_files_return_empty_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'missing.csv'
            empty = Path(directory) / 'empty.csv'
            empty.touch()

            self.assertTrue(csv_store.read_existing_csv(missing).empty)
            self.assertTrue(csv_store.read_existing_csv(empty).empty)

    def test_reads_utf8_csv_and_preserves_issue_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'draws.csv'
            path.write_text(
                '期号,日期,红球,蓝球\n'
                '2026001,2026-01-01,"01,02,03,04,05,06",07\n',
                encoding='utf-8',
            )

            result = csv_store.read_existing_csv(path)

            self.assertEqual(result.iloc[0]['期号'], '2026001')

    def test_atomic_write_replaces_file_without_temporary_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'draws.csv'
            frame = pd.DataFrame([{'期号': '2026001', '蓝球': '07'}])

            csv_store.atomic_write_csv(frame, path)

            result = pd.read_csv(path, dtype=str)
            self.assertEqual(result.to_dict('records'), [
                {'期号': '2026001', '蓝球': '07'},
            ])
            self.assertEqual(list(path.parent.glob('.ssq-*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
