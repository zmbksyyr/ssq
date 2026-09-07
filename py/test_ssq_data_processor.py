import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ssq_data_processor as processor
from ssq_core import DRAW_WEEKDAYS


class DataProcessorTests(unittest.TestCase):
    def test_http_session_retries_transient_statuses(self):
        session = processor.create_http_session()
        retry = session.get_adapter('https://').max_retries
        self.assertEqual(retry.total, 3)
        self.assertIn(429, retry.status_forcelist)

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

    def test_update_csv_validates_merges_and_sorts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'draws.csv'
            path.write_text(
                '期号,日期,红球,蓝球\n'
                '2026002,2026-01-04,"01,02,03,04,05,06",07\n',
                encoding='utf-8',
            )
            updated = processor.update_csv_file(path, [{
                '期号': '2026001', '日期': '2026-01-01',
                '红球': '6,5,4,3,2,1', '蓝球': '09',
            }])
            self.assertTrue(updated)
            result = pd.read_csv(path, dtype=str)
            self.assertEqual(result['期号'].tolist(), ['2026001', '2026002'])
            self.assertEqual(result.iloc[0]['红球'], '01,02,03,04,05,06')

    def test_update_csv_preserves_invalid_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'draws.csv'
            original = 'not,a,lottery,csv\nbad,row,data,here\n'
            path.write_text(original, encoding='utf-8')
            updated = processor.update_csv_file(path, [{
                '期号': '2026001', '日期': '2026-01-01',
                '红球': '1,2,3,4,5,6', '蓝球': '09',
            }])
            self.assertFalse(updated)
            self.assertEqual(path.read_text(encoding='utf-8'), original)

    def test_cross_check_sources_reports_mismatch(self):
        primary = [{'期号': '2026001', '红球': '01,02,03,04,05,06', '蓝球': '07'}]
        secondary = [{'期号': '2026001', '红球': '01,02,03,04,05,08', '蓝球': '07'}]
        self.assertEqual(processor.cross_check_sources(primary, secondary), ['2026001'])

    def test_cross_check_reports_issues_missing_from_authoritative_source(self):
        primary = [
            {'期号': '2026001', '红球': '01,02,03,04,05,06', '蓝球': '07'},
        ]
        secondary = [
            {'期号': '2026001', '红球': '01,02,03,04,05,06', '蓝球': '07'},
            {'期号': '2026002', '红球': '02,03,04,05,06,07', '蓝球': '08'},
        ]

        self.assertEqual(
            processor.find_secondary_only_issues(primary, secondary),
            ['2026002'],
        )

    def test_normalize_frame_rejects_issue_date_year_mismatch(self):
        frame = pd.DataFrame([{
            '期号': '2026001', '日期': '2025-12-31',
            '红球': '1,2,3,4,5,6', '蓝球': '07',
        }])
        with self.assertRaises(ValueError):
            processor.normalize_lottery_frame(frame)

    def test_normalize_frame_rejects_duplicate_issues(self):
        frame = pd.DataFrame([
            {'期号': '2026001', '日期': '2026-01-01', '红球': '1,2,3,4,5,6', '蓝球': '07'},
            {'期号': '2026001', '日期': '2026-01-01', '红球': '1,2,3,4,5,8', '蓝球': '09'},
        ])
        with self.assertRaises(ValueError):
            processor.normalize_lottery_frame(frame)

    def test_normalize_frame_rejects_date_order_mismatch(self):
        frame = pd.DataFrame([
            {'期号': '2026001', '日期': '2026-01-04', '红球': '1,2,3,4,5,6', '蓝球': '07'},
            {'期号': '2026002', '日期': '2026-01-01', '红球': '1,2,3,4,5,8', '蓝球': '09'},
        ])
        with self.assertRaisesRegex(ValueError, '日期顺序'):
            processor.normalize_lottery_frame(frame)

    def test_strict_snapshot_preserves_csv_when_download_is_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'draws.csv'
            original = (
                '期号,日期,红球,蓝球\n'
                '2026001,2026-01-01,"01,02,03,04,05,06",07\n'
            )
            path.write_text(original, encoding='utf-8')
            updated = processor.update_csv_file(
                path,
                [{
                    '期号': '2026001', '日期': '2026-01-01',
                    '红球': '1,2,3,4,5,6', '蓝球': '07',
                }],
                require_full_snapshot=True,
            )
            self.assertFalse(updated)
            self.assertEqual(path.read_text(encoding='utf-8'), original)

    def test_complete_authoritative_snapshot_is_accepted(self):
        current = date(2026, 1, 1)
        draw_dates = []
        while len(draw_dates) < processor.MIN_FULL_SNAPSHOT_RECORDS:
            if current.weekday() in DRAW_WEEKDAYS:
                draw_dates.append(current)
            current += timedelta(days=1)
        records = pd.DataFrame([
            {
                '期号': 2026001 + index,
                '日期': draw_date.isoformat(),
                '红球': '01,02,03,04,05,06',
                '蓝球': '07',
            }
            for index, draw_date in enumerate(draw_dates)
        ])
        normalized = processor.normalize_lottery_frame(records)
        processor.validate_authoritative_snapshot(
            normalized, normalized.iloc[:-1], today=date(2026, 12, 31)
        )

    def test_workflow_uses_txt_as_authority_and_reuses_session(self):
        records = [
            f'{2026001 + index} 2026-01-01 1 2 3 4 5 6 7'
            for index in range(processor.MIN_FULL_SNAPSHOT_RECORDS)
        ]
        session = object()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'draws.csv'
            with (
                patch(
                    'ssq_data_workflow.fetch_full_data_from_txt',
                    return_value=records,
                ) as fetch_txt,
                patch(
                    'ssq_data_workflow.fetch_latest_data_from_html',
                    return_value=[],
                ) as fetch_html,
            ):
                result = processor.run_data_update(path, session=session)

            self.assertEqual(result, str(path))
            self.assertTrue(path.exists())
            self.assertIs(fetch_txt.call_args.kwargs['session'], session)
            self.assertIs(fetch_html.call_args.kwargs['session'], session)

    def test_workflow_never_uses_html_without_authoritative_txt(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            'ssq_data_workflow.fetch_full_data_from_txt', return_value=[]
        ), patch(
            'ssq_data_workflow.fetch_latest_data_from_html'
        ) as fetch_html:
            with self.assertRaisesRegex(SystemExit, 'TXT 权威数据源'):
                processor.run_data_update(
                    Path(directory) / 'draws.csv', session=object()
                )
            fetch_html.assert_not_called()

    def test_workflow_rejects_txt_snapshot_missing_an_html_issue(self):
        records = [
            f'{2026001 + index} 2026-01-01 1 2 3 4 5 6 7'
            for index in range(processor.MIN_FULL_SNAPSHOT_RECORDS)
        ]
        html_records = [{
            '期号': '2026101',
            '红球': '01,02,03,04,05,06',
            '蓝球': '07',
        }]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                'ssq_data_workflow.fetch_full_data_from_txt',
                return_value=records,
            ),
            patch(
                'ssq_data_workflow.fetch_latest_data_from_html',
                return_value=html_records,
            ),
            patch('ssq_data_workflow.update_csv_file') as update,
        ):
            with self.assertRaisesRegex(SystemExit, 'TXT 权威源缺失'):
                processor.run_data_update(
                    Path(directory) / 'draws.csv', session=object()
                )
            update.assert_not_called()


if __name__ == '__main__':
    unittest.main()
