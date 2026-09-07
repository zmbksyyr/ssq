import io
import sys
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent))
import ssq_bonus_workflow as workflow
from ssq_domain import LOCAL_TIMEZONE


class BonusWorkflowTests(unittest.TestCase):
    def test_default_dependencies_bind_workflow_collaborators(self):
        dependencies = workflow.default_bonus_dependencies()

        self.assertIs(dependencies.load_draw, workflow.load_latest_draw)
        self.assertIs(dependencies.find_report, workflow.find_matching_report)
        self.assertIs(dependencies.parse_bets, workflow.parse_report_bets)
        self.assertIs(dependencies.format_report, workflow.format_bonus_report)
        self.assertIs(dependencies.write_text, workflow.atomic_write_text)
        self.assertIs(dependencies.now, workflow.local_now)

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
            workflow.default_bonus_dependencies(),
            load_draw=load_draw,
            find_report=find_report,
            parse_bets=parse_bets,
            format_report=format_report,
            write_text=write_text,
            now=now,
        )

        with patch('sys.stdout', new_callable=io.StringIO):
            result = workflow.run_bonus_check(
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
        self.assertIsInstance(report_data, workflow.BonusReportData)
        self.assertIs(report_data.single_bets, single_bets)
        self.assertIs(report_data.duplex_bet, duplex_bet)
        self.assertIs(report_data.generated_at, generated_at)
        write_text.assert_called_once_with(result, 'bonus report')

    def test_draw_loading_error_becomes_workflow_exit(self):
        find_report = Mock()
        dependencies = replace(
            workflow.default_bonus_dependencies(),
            load_draw=Mock(side_effect=ValueError('invalid history')),
            find_report=find_report,
        )

        with self.assertRaisesRegex(SystemExit, '读取 draws.csv 文件失败'):
            workflow.run_bonus_check(
                csv_path='draws.csv',
                dependencies=dependencies,
            )

        find_report.assert_not_called()


if __name__ == '__main__':
    unittest.main()
