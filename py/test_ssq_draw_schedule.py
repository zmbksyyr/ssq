import sys
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import ssq_core
import ssq_draw_schedule as schedule
from ssq_domain import LOCAL_TIMEZONE


class DrawScheduleTests(unittest.TestCase):
    def test_core_exports_schedule_functions(self):
        self.assertIs(ssq_core.local_now, schedule.local_now)
        self.assertIs(ssq_core.local_today, schedule.local_today)
        self.assertIs(ssq_core.infer_next_issue, schedule.infer_next_issue)

    def test_local_now_uses_shanghai_timezone(self):
        current = schedule.local_now()

        self.assertEqual(current.tzinfo, LOCAL_TIMEZONE)
        self.assertEqual(current.tzinfo.key, 'Asia/Shanghai')

    def test_local_today_uses_local_now(self):
        current = datetime(2026, 9, 7, 23, 59, tzinfo=LOCAL_TIMEZONE)

        with patch.object(schedule, 'local_now', return_value=current):
            self.assertEqual(schedule.local_today(), date(2026, 9, 7))

    def test_infers_next_issue_in_same_year(self):
        self.assertEqual(schedule.infer_next_issue(2026103, '2026-09-06'), 2026104)

    def test_infers_next_issue_across_year_boundary(self):
        self.assertEqual(schedule.infer_next_issue(2025153, '2025-12-30'), 2026001)

    def test_accepts_date_and_datetime_draw_dates(self):
        self.assertEqual(schedule.infer_next_issue(2026103, date(2026, 9, 6)), 2026104)
        self.assertEqual(
            schedule.infer_next_issue(
                2026103,
                datetime(2026, 9, 6, 21, 15, tzinfo=LOCAL_TIMEZONE),
            ),
            2026104,
        )

    def test_rejects_issue_date_year_mismatch(self):
        with self.assertRaisesRegex(ValueError, '期号年份 2025'):
            schedule.infer_next_issue(2025100, '2026-01-01')


if __name__ == '__main__':
    unittest.main()
