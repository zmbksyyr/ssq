import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ssq_data_store
import ssq_snapshot_validation as validation
from ssq_domain import DRAW_COLUMNS


class SnapshotValidationTests(unittest.TestCase):
    def test_data_store_preserves_validation_compatibility_exports(self):
        self.assertEqual(
            ssq_data_store.MIN_FULL_SNAPSHOT_RECORDS,
            validation.MIN_FULL_SNAPSHOT_RECORDS,
        )
        self.assertIs(
            ssq_data_store.normalize_lottery_frame,
            validation.normalize_lottery_frame,
        )
        self.assertIs(
            ssq_data_store.validate_authoritative_snapshot,
            validation.validate_authoritative_snapshot,
        )

    def test_rejects_snapshot_below_minimum_record_count(self):
        snapshot = pd.DataFrame(columns=DRAW_COLUMNS)

        with self.assertRaises(ValueError):
            validation.validate_authoritative_snapshot(
                snapshot,
                pd.DataFrame(columns=DRAW_COLUMNS),
            )

    def test_rejects_snapshot_with_future_draw_date(self):
        snapshot = pd.DataFrame({
            DRAW_COLUMNS[0]: range(validation.MIN_FULL_SNAPSHOT_RECORDS),
            DRAW_COLUMNS[1]: ['2026-01-04'] * validation.MIN_FULL_SNAPSHOT_RECORDS,
        })

        with self.assertRaises(ValueError):
            validation.validate_authoritative_snapshot(
                snapshot,
                pd.DataFrame(columns=DRAW_COLUMNS),
                today=date(2026, 1, 3),
            )

    def test_rejects_snapshot_missing_existing_issue(self):
        snapshot = pd.DataFrame({
            DRAW_COLUMNS[0]: range(validation.MIN_FULL_SNAPSHOT_RECORDS),
            DRAW_COLUMNS[1]: ['2026-01-01'] * validation.MIN_FULL_SNAPSHOT_RECORDS,
        })
        existing = pd.DataFrame({DRAW_COLUMNS[0]: [999]})

        with self.assertRaisesRegex(ValueError, '999'):
            validation.validate_authoritative_snapshot(
                snapshot,
                existing,
                today=date(2026, 1, 1),
            )


if __name__ == '__main__':
    unittest.main()
