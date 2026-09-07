import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ssq_draw_data import (
    normalize_draw_frame,
    serialize_draw_frame,
    validate_draw_dates_not_future,
)


class DrawDataTests(unittest.TestCase):
    def test_normalizes_column_order_rows_and_value_types(self):
        frame = pd.DataFrame({
            '蓝球': ['08', '07'],
            '红球': ['7,6,5,4,3,2', '6,5,4,3,2,1'],
            '日期': ['2026-01-04', '2026-01-01'],
            '期号': ['2026002', '2026001'],
        })

        result = normalize_draw_frame(frame)

        self.assertEqual(tuple(result.columns), ('期号', '日期', '红球', '蓝球'))
        self.assertEqual(result['期号'].tolist(), [2026001, 2026002])
        self.assertEqual(result.iloc[0]['红球'], [1, 2, 3, 4, 5, 6])
        self.assertEqual(result.iloc[1]['蓝球'], 8)

    def test_rejects_missing_and_extra_columns(self):
        for columns in (
            {'期号': [], '日期': [], '红球': []},
            {'期号': [], '日期': [], '红球': [], '蓝球': [], '备注': []},
        ):
            with self.subTest(columns=columns), self.assertRaisesRegex(
                ValueError, '字段不匹配'
            ):
                normalize_draw_frame(pd.DataFrame(columns))

    def test_serializes_numbers_with_stable_padding(self):
        frame = pd.DataFrame([{
            '期号': 2026001,
            '日期': '2026-01-01',
            '红球': [6, 5, 4, 3, 2, 1],
            '蓝球': 7,
        }])

        result = serialize_draw_frame(frame)

        self.assertEqual(result.iloc[0]['红球'], '01,02,03,04,05,06')
        self.assertEqual(result.iloc[0]['蓝球'], '07')

    def test_rejects_dates_outside_the_draw_calendar(self):
        frame = pd.DataFrame([{
            '期号': 2026001,
            '日期': '2026-01-02',
            '红球': '01,02,03,04,05,06',
            '蓝球': '07',
        }])

        with self.assertRaisesRegex(ValueError, '周二、周四或周日'):
            normalize_draw_frame(frame)

    def test_rejects_multiple_issues_on_the_same_draw_date(self):
        frame = pd.DataFrame([
            {
                '期号': 2026001,
                '日期': '2026-01-01',
                '红球': '01,02,03,04,05,06',
                '蓝球': '07',
            },
            {
                '期号': 2026002,
                '日期': '2026-01-01',
                '红球': '02,03,04,05,06,07',
                '蓝球': '08',
            },
        ])

        with self.assertRaisesRegex(ValueError, '同一开奖日期'):
            normalize_draw_frame(frame)

    def test_rejects_issue_gaps_and_invalid_year_resets(self):
        cases = (
            (2026001, '2026-01-01', 2026003, '2026-01-06'),
            (2026153, '2026-12-31', 2027002, '2027-01-05'),
            (2026153, '2026-12-31', 2028001, '2028-01-02'),
        )
        for first_issue, first_date, second_issue, second_date in cases:
            frame = pd.DataFrame([
                {
                    '期号': first_issue,
                    '日期': first_date,
                    '红球': '01,02,03,04,05,06',
                    '蓝球': '07',
                },
                {
                    '期号': second_issue,
                    '日期': second_date,
                    '红球': '02,03,04,05,06,07',
                    '蓝球': '08',
                },
            ])
            with self.subTest(issues=(first_issue, second_issue)), (
                self.assertRaisesRegex(ValueError, '期号不连续')
            ):
                normalize_draw_frame(frame)

    def test_current_workflows_reject_future_draw_dates(self):
        frame = pd.DataFrame({'日期': ['2026-01-01', '2026-01-04']})
        validate_draw_dates_not_future(frame, today=pd.Timestamp('2026-01-04').date())

        with self.assertRaisesRegex(ValueError, '未来开奖日期'):
            validate_draw_dates_not_future(
                frame,
                today=pd.Timestamp('2026-01-03').date(),
            )


if __name__ == '__main__':
    unittest.main()
