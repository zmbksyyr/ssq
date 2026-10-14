"""Consolidated data_prep tests (merged from 15 module test files).

Originally one test module per production module. The production modules are now
merged into py/ssq_data_prep.py, so the tests are grouped the same way. Retired module
names still resolve through ssq_compat, which run_tests.py sets up before these
modules are imported.

Run with: python py/run_tests.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ast
import io
import sys
import tempfile
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy
import pandas
import ssq_ball_scoring as ball_scoring
import ssq_bonus_workflow as bonus_workflow
import ssq_core
import ssq_core as core
import ssq_csv_store as csv_store
import ssq_data_processor as data_processor
import ssq_data_sources as data_sources
import ssq_data_store
import ssq_data_workflow as data_workflow
import ssq_domain as domain
import ssq_draw_schedule as draw_schedule
import ssq_file_io as file_io
import ssq_history_store as history_store
import ssq_http as http
import ssq_parsing as parsing
import ssq_paths as paths
import ssq_snapshot_validation as snapshot_validation
import ssq_source_comparison as source_comparison
import ssq_source_parsing as source_parsing
from ssq_core import DRAW_WEEKDAYS
from ssq_domain import BLUE_BALLS, DRAW_COLUMNS, LOCAL_TIMEZONE, RED_BALLS
from ssq_draw_data import (
    fingerprint_draw_frame,
    normalize_draw_frame,
    serialize_draw_frame,
    validate_draw_dates_not_future,
)

# ===== from test_ssq_ball_scoring.py =====


class BallScoringTests(unittest.TestCase):
    def test_fusion_uses_injected_calculation_boundaries(self):
        params = {
            'decay_factor': 0.9,
            'weight_freq': 0.4,
            'weight_omission': 0.5,
            'weight_ml': 0.1,
            'weight_blue_freq': 0.6,
            'weight_blue_ml': 0.4,
        }
        history = pandas.DataFrame({
            '红球': [[1, 2, 3, 4, 5, 6]],
            '蓝球': [1],
            'feature': [0.0],
        })
        red_models = {ball: object() for ball in RED_BALLS}
        blue_models = {ball: object() for ball in BLUE_BALLS}
        dependencies = ball_scoring.BallScoringDependencies(
            validate_params=Mock(return_value=params),
            validate_models=Mock(),
            validate_features=Mock(return_value=('feature',)),
            predict_probability=Mock(return_value=numpy.array([1.0])),
            weighted_frequency=Mock(side_effect=(
                pandas.Series({ball: 1.0 for ball in RED_BALLS}),
                pandas.Series({ball: 1.0 for ball in BLUE_BALLS}),
            )),
            omission=Mock(return_value={ball: 1 for ball in RED_BALLS}),
            adjust_red_scores=Mock(side_effect=lambda scores, *_: scores),
            validate_scores=Mock(side_effect=lambda scores, *_: scores),
        )

        red_scores, blue_scores = ball_scoring.run_strategy_and_get_scores(
            history,
            params,
            red_models,
            blue_models,
            ('feature',),
            dependencies,
        )

        self.assertTrue(all(score == 1.0 for score in red_scores.values()))
        self.assertTrue(all(score == 1.0 for score in blue_scores.values()))
        dependencies.validate_models.assert_called_once_with(
            red_models,
            blue_models,
        )
        self.assertEqual(dependencies.weighted_frequency.call_count, 2)

    def test_production_workflows_do_not_import_scoring_facade(self):
        filenames = (
            'ssq_backtesting.py',
            'ssq_backtest_preparation.py',
            'ssq_prediction_workflow.py',
            'ssq_workflow.py',
        )
        for filename in filenames:
            tree = ast.parse(
                Path(__file__).with_name(filename).read_text(encoding='utf-8')
            )
            imports = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            with self.subTest(filename=filename):
                self.assertNotIn('ssq_scoring', imports)


# ===== from test_ssq_bonus_workflow.py =====


class BonusWorkflowTests(unittest.TestCase):
    def test_default_dependencies_bind_workflow_collaborators(self):
        dependencies = bonus_workflow.default_bonus_dependencies()

        self.assertIs(dependencies.load_draw, bonus_workflow.load_latest_draw)
        self.assertIs(dependencies.find_report, bonus_workflow.find_matching_report)
        self.assertIs(dependencies.parse_bets, bonus_workflow.parse_report_bets)
        self.assertIs(dependencies.format_report, bonus_workflow.format_bonus_report)
        self.assertIs(dependencies.write_text, bonus_workflow.atomic_write_text)
        self.assertIs(dependencies.now, bonus_workflow.local_now)

    def test_run_composes_dependencies_and_uses_one_generation_time(self):
        generated_at = datetime(
            2026, 9, 8, 1, 2, 3, tzinfo=LOCAL_TIMEZONE
        )
        latest_draw = {
            'issue': 2026104,
            'red': {1, 2, 3, 4, 5, 6},
            'blue': 7,
        }
        single_bets = [{'red': [1, 2, 3, 4, 5, 6], 'blue': 7}]
        duplex_bet = {'red': [1, 2, 3, 4, 5, 6, 7], 'blue': [7]}
        load_draw = Mock(return_value=latest_draw)
        find_report = Mock(return_value=('analysis.txt', None))
        parse_bets = Mock(return_value=(single_bets, duplex_bet))
        format_report = Mock(return_value='bonus report')
        write_text = Mock()
        now = Mock(return_value=generated_at)
        dependencies = replace(
            bonus_workflow.default_bonus_dependencies(),
            load_draw=load_draw,
            find_report=find_report,
            parse_bets=parse_bets,
            format_report=format_report,
            write_text=write_text,
            now=now,
        )

        with patch('sys.stdout', new_callable=io.StringIO):
            result = bonus_workflow.run_bonus_check(
                csv_path='draws.csv',
                report_dir='reports',
                dependencies=dependencies,
            )

        self.assertEqual(Path(result).name, 'ssq_bonus_check_20260908_010203.txt')
        load_draw.assert_called_once_with('draws.csv')
        find_report.assert_called_once_with(2026104, 'reports')
        parse_bets.assert_called_once_with('analysis.txt')
        now.assert_called_once_with()
        report_data = format_report.call_args.args[0]
        self.assertIsInstance(report_data, bonus_workflow.BonusReportData)
        self.assertIs(report_data.single_bets, single_bets)
        self.assertIs(report_data.duplex_bet, duplex_bet)
        self.assertIs(report_data.generated_at, generated_at)
        write_text.assert_called_once_with(result, 'bonus report')

    def test_draw_loading_error_becomes_workflow_exit(self):
        find_report = Mock()
        dependencies = replace(
            bonus_workflow.default_bonus_dependencies(),
            load_draw=Mock(side_effect=ValueError('invalid history')),
            find_report=find_report,
        )

        with self.assertRaisesRegex(SystemExit, '读取 draws.csv 文件失败'):
            bonus_workflow.run_bonus_check(
                csv_path='draws.csv',
                dependencies=dependencies,
            )

        find_report.assert_not_called()


# ===== from test_ssq_csv_store.py =====


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
            frame = pandas.DataFrame([{'期号': '2026001', '蓝球': '07'}])

            csv_store.atomic_write_csv(frame, path)

            result = pandas.read_csv(path, dtype=str)
            self.assertEqual(result.to_dict('records'), [
                {'期号': '2026001', '蓝球': '07'},
            ])
            self.assertEqual(list(path.parent.glob('.ssq-*.tmp')), [])


# ===== from test_ssq_data_processor.py =====


class DataProcessorTests(unittest.TestCase):
    def test_http_session_retries_transient_statuses(self):
        session = data_processor.create_http_session()
        retry = session.get_adapter('https://').max_retries
        self.assertEqual(retry.total, 3)
        self.assertIn(429, retry.status_forcelist)

    def test_parse_txt_data_validates_and_normalizes(self):
        rows = data_processor.parse_txt_data([
            '2026001 2026-01-01 6 1 10 3 8 2 9',
            '2026002 2026-01-03 1 1 2 3 4 5 6',
            '2026003 invalid 1 2 3 4 5 6 7',
            '2026004 2026-01-05 1 2 3 4 5 34 7',
        ])
        self.assertEqual(rows, [[
            '2026001', '2026-01-01', '01,02,03,06,08,10', '09'
        ]])

    def test_html_parser_uses_shared_issue_and_ball_validation(self):
        response = SimpleNamespace(
            text=(
                '<table><tr><th>期号</th><th>红球</th><th>蓝球</th></tr>'
                '<tr><td>2026001期</td><td><i>6</i>  <i>1</i>\n<i>10</i> '
                '<i>3</i> <i>8</i> <i>2</i></td><td>9</td></tr>'
                '<tr><td>999期</td><td>1 2 3 4 5 6</td><td>7</td></tr>'
                '</table>'
            ),
            raise_for_status=lambda: None,
        )
        session = SimpleNamespace(get=lambda *_args, **_kwargs: response)

        records = data_processor.fetch_latest_data_from_html(session=session)

        self.assertEqual(records, [{
            '期号': '2026001',
            '红球': '01,02,03,06,08,10',
            '蓝球': '09',
        }])

    def test_update_csv_validates_merges_and_sorts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'draws.csv'
            path.write_text(
                '期号,日期,红球,蓝球\n'
                '2026002,2026-01-04,"01,02,03,04,05,06",07\n',
                encoding='utf-8',
            )
            updated = data_processor.update_csv_file(path, [{
                '期号': '2026001', '日期': '2026-01-01',
                '红球': '6,5,4,3,2,1', '蓝球': '09',
            }])
            self.assertTrue(updated)
            result = pandas.read_csv(path, dtype=str)
            self.assertEqual(result['期号'].tolist(), ['2026001', '2026002'])
            self.assertEqual(result.iloc[0]['红球'], '01,02,03,04,05,06')

    def test_update_csv_preserves_invalid_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'draws.csv'
            original = 'not,a,lottery,csv\nbad,row,data,here\n'
            path.write_text(original, encoding='utf-8')
            updated = data_processor.update_csv_file(path, [{
                '期号': '2026001', '日期': '2026-01-01',
                '红球': '1,2,3,4,5,6', '蓝球': '09',
            }])
            self.assertFalse(updated)
            self.assertEqual(path.read_text(encoding='utf-8'), original)

    def test_cross_check_sources_reports_mismatch(self):
        primary = [{'期号': '2026001', '红球': '01,02,03,04,05,06', '蓝球': '07'}]
        secondary = [{'期号': '2026001', '红球': '01,02,03,04,05,08', '蓝球': '07'}]
        self.assertEqual(data_processor.cross_check_sources(primary, secondary), ['2026001'])

    def test_cross_check_reports_issues_missing_from_authoritative_source(self):
        primary = [
            {'期号': '2026001', '红球': '01,02,03,04,05,06', '蓝球': '07'},
        ]
        secondary = [
            {'期号': '2026001', '红球': '01,02,03,04,05,06', '蓝球': '07'},
            {'期号': '2026002', '红球': '02,03,04,05,06,07', '蓝球': '08'},
        ]

        self.assertEqual(
            data_processor.find_secondary_only_issues(primary, secondary),
            ['2026002'],
        )

    def test_normalize_frame_rejects_issue_date_year_mismatch(self):
        frame = pandas.DataFrame([{
            '期号': '2026001', '日期': '2025-12-31',
            '红球': '1,2,3,4,5,6', '蓝球': '07',
        }])
        with self.assertRaises(ValueError):
            data_processor.normalize_lottery_frame(frame)

    def test_normalize_frame_rejects_duplicate_issues(self):
        frame = pandas.DataFrame([
            {'期号': '2026001', '日期': '2026-01-01', '红球': '1,2,3,4,5,6', '蓝球': '07'},
            {'期号': '2026001', '日期': '2026-01-01', '红球': '1,2,3,4,5,8', '蓝球': '09'},
        ])
        with self.assertRaises(ValueError):
            data_processor.normalize_lottery_frame(frame)

    def test_normalize_frame_rejects_date_order_mismatch(self):
        frame = pandas.DataFrame([
            {'期号': '2026001', '日期': '2026-01-04', '红球': '1,2,3,4,5,6', '蓝球': '07'},
            {'期号': '2026002', '日期': '2026-01-01', '红球': '1,2,3,4,5,8', '蓝球': '09'},
        ])
        with self.assertRaisesRegex(ValueError, '日期顺序'):
            data_processor.normalize_lottery_frame(frame)

    def test_strict_snapshot_preserves_csv_when_download_is_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'draws.csv'
            original = (
                '期号,日期,红球,蓝球\n'
                '2026001,2026-01-01,"01,02,03,04,05,06",07\n'
            )
            path.write_text(original, encoding='utf-8')
            updated = data_processor.update_csv_file(
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
        while len(draw_dates) < data_processor.MIN_FULL_SNAPSHOT_RECORDS:
            if current.weekday() in DRAW_WEEKDAYS:
                draw_dates.append(current)
            current += timedelta(days=1)
        records = pandas.DataFrame([
            {
                '期号': 2026001 + index,
                '日期': draw_date.isoformat(),
                '红球': '01,02,03,04,05,06',
                '蓝球': '07',
            }
            for index, draw_date in enumerate(draw_dates)
        ])
        normalized = data_processor.normalize_lottery_frame(records)
        data_processor.validate_authoritative_snapshot(
            normalized, normalized.iloc[:-1], today=date(2026, 12, 31)
        )

    def test_workflow_uses_txt_as_authority_and_reuses_session(self):
        current = date(2026, 1, 1)
        draw_dates = []
        while len(draw_dates) < data_processor.MIN_FULL_SNAPSHOT_RECORDS:
            if current.weekday() in DRAW_WEEKDAYS:
                draw_dates.append(current)
            current += timedelta(days=1)
        records = [
            f'{2026001 + index} {draw_date.isoformat()} 1 2 3 4 5 6 7'
            for index, draw_date in enumerate(draw_dates)
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
                result = data_processor.run_data_update(path, session=session)

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
                data_processor.run_data_update(
                    Path(directory) / 'draws.csv', session=object()
                )
            fetch_html.assert_not_called()

    def test_workflow_rejects_txt_snapshot_missing_an_html_issue(self):
        records = [
            f'{2026001 + index} 2026-01-01 1 2 3 4 5 6 7'
            for index in range(data_processor.MIN_FULL_SNAPSHOT_RECORDS)
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
                data_processor.run_data_update(
                    Path(directory) / 'draws.csv', session=object()
                )
            update.assert_not_called()


# ===== from test_ssq_domain.py =====


class DomainTests(unittest.TestCase):
    def test_core_preserves_domain_constant_exports(self):
        for name in (
            'RED_BALLS',
            'BLUE_BALLS',
            'PRIME_RED_BALLS',
            'DRAW_COLUMNS',
            'DRAW_WEEKDAYS',
            'LOCAL_TIMEZONE',
            'PRIZE_RULES',
            'PRIZE_NAMES',
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(domain, name))

    def test_lottery_domains_and_schedule_are_complete(self):
        self.assertEqual(domain.RED_BALLS, tuple(range(1, 34)))
        self.assertEqual(domain.BLUE_BALLS, tuple(range(1, 17)))
        self.assertEqual(domain.DRAW_WEEKDAYS, {1, 3, 6})
        self.assertEqual(domain.PRIZE_RULES[(6, 1)], 5_000_000)


# ===== from test_ssq_draw_data.py =====


class DrawDataTests(unittest.TestCase):
    def test_normalizes_column_order_rows_and_value_types(self):
        frame = pandas.DataFrame({
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
                normalize_draw_frame(pandas.DataFrame(columns))

    def test_serializes_numbers_with_stable_padding(self):
        frame = pandas.DataFrame([{
            '期号': 2026001,
            '日期': '2026-01-01',
            '红球': [6, 5, 4, 3, 2, 1],
            '蓝球': 7,
        }])

        result = serialize_draw_frame(frame)

        self.assertEqual(result.iloc[0]['红球'], '01,02,03,04,05,06')
        self.assertEqual(result.iloc[0]['蓝球'], '07')

    def test_history_fingerprint_is_canonical_and_content_sensitive(self):
        first = pandas.DataFrame({
            '期号': ['2026002', '2026001'],
            '日期': ['2026-01-04', '2026-01-01'],
            '红球': ['7,6,5,4,3,2', '6,5,4,3,2,1'],
            '蓝球': ['08', '07'],
        })
        equivalent = pandas.DataFrame({
            '蓝球': [7, 8],
            '红球': [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]],
            '日期': ['2026-01-01', '2026-01-04'],
            '期号': [2026001, 2026002],
        })
        changed = equivalent.copy(deep=True)
        changed.at[1, '红球'] = [2, 3, 4, 5, 6, 8]

        self.assertEqual(
            fingerprint_draw_frame(first),
            fingerprint_draw_frame(equivalent),
        )
        self.assertNotEqual(
            fingerprint_draw_frame(equivalent),
            fingerprint_draw_frame(changed),
        )

    def test_rejects_dates_outside_the_draw_calendar(self):
        frame = pandas.DataFrame([{
            '期号': 2026001,
            '日期': '2026-01-02',
            '红球': '01,02,03,04,05,06',
            '蓝球': '07',
        }])

        with self.assertRaisesRegex(ValueError, '周二、周四或周日'):
            normalize_draw_frame(frame)

    def test_rejects_multiple_issues_on_the_same_draw_date(self):
        frame = pandas.DataFrame([
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
            frame = pandas.DataFrame([
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
        frame = pandas.DataFrame({'日期': ['2026-01-01', '2026-01-04']})
        validate_draw_dates_not_future(frame, today=pandas.Timestamp('2026-01-04').date())

        with self.assertRaisesRegex(ValueError, '未来开奖日期'):
            validate_draw_dates_not_future(
                frame,
                today=pandas.Timestamp('2026-01-03').date(),
            )


# ===== from test_ssq_draw_schedule.py =====


class DrawScheduleTests(unittest.TestCase):
    def test_core_exports_schedule_functions(self):
        self.assertIs(ssq_core.local_now, draw_schedule.local_now)
        self.assertIs(ssq_core.local_today, draw_schedule.local_today)
        self.assertIs(ssq_core.infer_next_issue, draw_schedule.infer_next_issue)

    def test_local_now_uses_shanghai_timezone(self):
        current = draw_schedule.local_now()

        self.assertEqual(current.tzinfo, LOCAL_TIMEZONE)
        self.assertEqual(current.tzinfo.key, 'Asia/Shanghai')

    def test_local_today_uses_local_now(self):
        current = datetime(2026, 9, 7, 23, 59, tzinfo=LOCAL_TIMEZONE)

        with patch.object(draw_schedule, 'local_now', return_value=current):
            self.assertEqual(draw_schedule.local_today(), date(2026, 9, 7))

    def test_infers_next_issue_in_same_year(self):
        self.assertEqual(draw_schedule.infer_next_issue(2026103, '2026-09-06'), 2026104)

    def test_infers_next_issue_across_year_boundary(self):
        self.assertEqual(draw_schedule.infer_next_issue(2025153, '2025-12-30'), 2026001)

    def test_accepts_date_and_datetime_draw_dates(self):
        self.assertEqual(draw_schedule.infer_next_issue(2026103, date(2026, 9, 6)), 2026104)
        self.assertEqual(
            draw_schedule.infer_next_issue(
                2026103,
                datetime(2026, 9, 6, 21, 15, tzinfo=LOCAL_TIMEZONE),
            ),
            2026104,
        )

    def test_rejects_issue_date_year_mismatch(self):
        with self.assertRaisesRegex(ValueError, '期号年份 2025'):
            draw_schedule.infer_next_issue(2025100, '2026-01-01')


# ===== from test_ssq_file_io.py =====


class AtomicFileIoTests(unittest.TestCase):
    def test_core_preserves_atomic_write_compatibility_export(self):
        self.assertIs(ssq_core.atomic_write_text, file_io.atomic_write_text)

    def test_atomic_write_creates_parent_and_replaces_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'report.txt'

            file_io.atomic_write_text(path, '新内容')

            self.assertEqual(path.read_text(encoding='utf-8'), '新内容')
            self.assertEqual(list(path.parent.glob('.ssq-*.tmp')), [])

    def test_writer_failure_preserves_target_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.txt'
            path.write_text('原内容', encoding='utf-8')

            with (
                self.assertRaisesRegex(RuntimeError, 'write failed'),
                file_io.atomic_text_writer(path) as temporary_file,
            ):
                temporary_file.write('不完整内容')
                raise RuntimeError('write failed')

            self.assertEqual(path.read_text(encoding='utf-8'), '原内容')
            self.assertEqual(list(path.parent.glob('.ssq-*.tmp')), [])


# ===== from test_ssq_history_store.py =====


class HistoryStoreTests(unittest.TestCase):
    def test_data_store_preserves_history_compatibility_export(self):
        self.assertIs(ssq_data_store.update_csv_file, history_store.update_csv_file)

    def test_merge_workflow_uses_injected_dependencies(self):
        normalize = Mock(side_effect=lambda frame: frame)
        read = Mock(return_value=pandas.DataFrame())
        validate = Mock()
        write = Mock()
        dependencies = history_store.HistoryStoreDependencies(
            normalize_frame=normalize,
            read_csv=read,
            validate_snapshot=validate,
            write_csv=write,
        )
        records = [{
            '期号': '2026001',
            '日期': '2026-01-01',
            '红球': '01,02,03,04,05,06',
            '蓝球': '07',
        }]

        updated = history_store.update_csv_file(
            'draws.csv',
            records,
            require_full_snapshot=True,
            dependencies=dependencies,
        )

        self.assertTrue(updated)
        read.assert_called_once_with('draws.csv')
        validate.assert_called_once()
        self.assertEqual(normalize.call_count, 2)
        written_frame, written_path = write.call_args.args
        self.assertEqual(written_frame.to_dict('records'), records)
        self.assertEqual(written_path, 'draws.csv')

    def test_production_modules_do_not_depend_on_compatibility_facade(self):
        for filename in ('ssq_data_workflow.py', 'ssq_data_processor.py'):
            path = Path(__file__).with_name(filename)
            tree = ast.parse(path.read_text(encoding='utf-8'))
            imports = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            with self.subTest(filename=filename):
                self.assertNotIn('ssq_data_store', imports)


# ===== from test_ssq_http.py =====


class HttpTransportTests(unittest.TestCase):
    def test_data_sources_preserves_http_compatibility_exports(self):
        self.assertIs(data_sources.create_http_session, http.create_http_session)
        self.assertIs(data_sources.fetch_text, http.fetch_text)
        self.assertIs(data_sources.REQUEST_HEADERS, http.REQUEST_HEADERS)
        self.assertEqual(
            data_sources.REQUEST_TIMEOUT_SECONDS,
            http.REQUEST_TIMEOUT_SECONDS,
        )

    def test_session_retries_transient_get_failures(self):
        session = http.create_http_session()
        retry = session.get_adapter('https://').max_retries

        self.assertEqual(retry.total, 3)
        self.assertEqual(tuple(retry.allowed_methods), ('GET',))
        self.assertIn(429, retry.status_forcelist)

    def test_fetch_text_applies_request_policy_and_encoding(self):
        response = SimpleNamespace(
            text='draw data',
            encoding=None,
            raise_for_status=Mock(),
        )
        session = SimpleNamespace(get=Mock(return_value=response))

        content = http.fetch_text(
            'https://example.test/draws.txt',
            session=session,
            encoding='utf-8',
        )

        self.assertEqual(content, 'draw data')
        self.assertEqual(response.encoding, 'utf-8')
        response.raise_for_status.assert_called_once_with()
        session.get.assert_called_once_with(
            'https://example.test/draws.txt',
            headers=http.REQUEST_HEADERS,
            timeout=http.REQUEST_TIMEOUT_SECONDS,
        )


# ===== from test_ssq_parsing.py =====


class ParsingTests(unittest.TestCase):
    def test_core_preserves_parsing_compatibility_exports(self):
        for name in (
            'parse_integer',
            'parse_issue',
            'parse_red_balls',
            'parse_blue_ball',
            'parse_blue_balls',
            'validate_ball_scores',
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(parsing, name))

    def test_score_validation_normalizes_complete_domain(self):
        scores = parsing.validate_ball_scores(
            {ball: ball for ball in RED_BALLS},
            RED_BALLS,
            '红球',
        )

        self.assertEqual(set(scores), set(RED_BALLS))
        self.assertTrue(all(type(value) is float for value in scores.values()))

    def test_score_validation_rejects_missing_and_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, '评分键不完整'):
            parsing.validate_ball_scores({1: 0.5}, RED_BALLS, '红球')
        with self.assertRaisesRegex(ValueError, '有限数值'):
            parsing.validate_ball_scores(
                {ball: float('nan') for ball in RED_BALLS},
                RED_BALLS,
                '红球',
            )


# ===== from test_ssq_paths.py =====


class ProjectPathTests(unittest.TestCase):
    def test_project_root_is_derived_from_the_source_directory(self):
        expected_root = Path(__file__).resolve().parent.parent

        self.assertEqual(Path(paths.SCRIPT_DIR), expected_root / 'py')
        self.assertEqual(Path(paths.PROJECT_ROOT), expected_root)
        self.assertEqual(Path(paths.CSV_PATH), expected_root / 'shuangseqiu.csv')
        self.assertEqual(Path(paths.PARAMS_JSON_PATH), expected_root / 'best_params.json')
        self.assertEqual(Path(paths.REPORT_DIR), expected_root / 'report')

    def test_workflows_share_canonical_path_objects(self):
        self.assertIs(bonus_workflow.PROJECT_ROOT, paths.PROJECT_ROOT)
        self.assertIs(bonus_workflow.CSV_PATH, paths.CSV_PATH)
        self.assertIs(bonus_workflow.PARAMS_JSON_PATH, paths.PARAMS_JSON_PATH)
        self.assertIs(bonus_workflow.REPORT_DIR, paths.REPORT_DIR)
        self.assertIs(bonus_workflow.PROJECT_ROOT, paths.PROJECT_ROOT)
        self.assertIs(bonus_workflow.CSV_PATH, paths.CSV_PATH)
        self.assertIs(bonus_workflow.REPORT_DIR, paths.REPORT_DIR)
        self.assertIs(data_workflow.PROJECT_ROOT, paths.PROJECT_ROOT)
        self.assertIs(data_workflow.CSV_FILE_PATH, paths.CSV_PATH)


# ===== from test_ssq_snapshot_validation.py =====


class SnapshotValidationTests(unittest.TestCase):
    def test_data_store_preserves_validation_compatibility_exports(self):
        self.assertEqual(
            ssq_data_store.MIN_FULL_SNAPSHOT_RECORDS,
            snapshot_validation.MIN_FULL_SNAPSHOT_RECORDS,
        )
        self.assertIs(
            ssq_data_store.normalize_lottery_frame,
            snapshot_validation.normalize_lottery_frame,
        )
        self.assertIs(
            ssq_data_store.validate_authoritative_snapshot,
            snapshot_validation.validate_authoritative_snapshot,
        )

    def test_rejects_snapshot_below_minimum_record_count(self):
        snapshot = pandas.DataFrame(columns=DRAW_COLUMNS)

        with self.assertRaises(ValueError):
            snapshot_validation.validate_authoritative_snapshot(
                snapshot,
                pandas.DataFrame(columns=DRAW_COLUMNS),
            )

    def test_rejects_snapshot_with_future_draw_date(self):
        snapshot = pandas.DataFrame({
            DRAW_COLUMNS[0]: range(snapshot_validation.MIN_FULL_SNAPSHOT_RECORDS),
            DRAW_COLUMNS[1]: ['2026-01-04'] * snapshot_validation.MIN_FULL_SNAPSHOT_RECORDS,
        })

        with self.assertRaises(ValueError):
            snapshot_validation.validate_authoritative_snapshot(
                snapshot,
                pandas.DataFrame(columns=DRAW_COLUMNS),
                today=date(2026, 1, 3),
            )

    def test_rejects_snapshot_missing_existing_issue(self):
        snapshot = pandas.DataFrame({
            DRAW_COLUMNS[0]: range(snapshot_validation.MIN_FULL_SNAPSHOT_RECORDS),
            DRAW_COLUMNS[1]: ['2026-01-01'] * snapshot_validation.MIN_FULL_SNAPSHOT_RECORDS,
        })
        existing = pandas.DataFrame({DRAW_COLUMNS[0]: [999]})

        with self.assertRaisesRegex(ValueError, '999'):
            snapshot_validation.validate_authoritative_snapshot(
                snapshot,
                existing,
                today=date(2026, 1, 1),
            )


# ===== from test_ssq_source_comparison.py =====


class SourceComparisonTests(unittest.TestCase):
    def test_data_sources_preserves_comparison_compatibility_exports(self):
        self.assertIs(data_sources.cross_check_sources, source_comparison.cross_check_sources)
        self.assertIs(
            data_sources.find_secondary_only_issues,
            source_comparison.find_secondary_only_issues,
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
            source_comparison.cross_check_sources(primary, secondary),
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
            source_comparison.find_secondary_only_issues(primary, secondary),
            ['2026002', '2026010'],
        )


# ===== from test_ssq_source_parsing.py =====


class SourceParsingTests(unittest.TestCase):
    def test_data_sources_preserves_parser_compatibility_exports(self):
        self.assertIs(data_sources.parse_html_data, source_parsing.parse_html_data)
        self.assertIs(data_sources.parse_txt_data, source_parsing.parse_txt_data)

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


if __name__ == "__main__":
    unittest.main()
