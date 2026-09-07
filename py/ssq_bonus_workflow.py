"""Application workflow for checking recommendations against a draw."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import ssq_latest_draw as _latest_draw
import ssq_report_discovery as _report_discovery
from pandas.errors import ParserError
from ssq_bet_parsing import parse_report_bets
from ssq_bonus_reporting import BonusReportData, format_bonus_report
from ssq_draw_data import normalize_draw_frame, validate_draw_dates_not_future
from ssq_draw_schedule import local_now
from ssq_file_io import atomic_write_text

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(PROJECT_ROOT, 'shuangseqiu.csv')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')
TARGET_ISSUE_PATTERN = _report_discovery.TARGET_ISSUE_PATTERN
parse_report_target_issue = _report_discovery.parse_report_target_issue


@dataclass(frozen=True)
class BonusCheckDependencies:
    load_draw: Callable[..., Any]
    find_report: Callable[..., Any]
    parse_bets: Callable[..., Any]
    format_report: Callable[..., str]
    write_text: Callable[..., Any]
    now: Callable[..., Any]


def find_matching_report(target_issue, report_dir=REPORT_DIR):
    """Compatibility wrapper using the default report directory."""
    return _report_discovery.find_matching_report(target_issue, report_dir)


def load_latest_draw(filepath=CSV_PATH):
    """Compatibility wrapper using workflow-level validation dependencies."""
    return _latest_draw.load_latest_draw(
        filepath,
        normalize=normalize_draw_frame,
        validate_dates=validate_draw_dates_not_future,
    )


def default_bonus_dependencies():
    return BonusCheckDependencies(
        load_draw=load_latest_draw,
        find_report=find_matching_report,
        parse_bets=parse_report_bets,
        format_report=format_bonus_report,
        write_text=atomic_write_text,
        now=local_now,
    )


def run_bonus_check(
    csv_path=CSV_PATH,
    report_dir=REPORT_DIR,
    generated_at=None,
    dependencies=None,
):
    """Run a complete prize check and return the generated report path."""
    dependencies = dependencies or default_bonus_dependencies()
    try:
        latest_draw = dependencies.load_draw(csv_path)
    except (
        OSError, UnicodeError, TypeError, ValueError, ParserError
    ) as exc:
        raise SystemExit(f'读取 {csv_path} 文件失败: {exc}') from exc

    target_issue = latest_draw['issue']
    report_filepath, error_message = dependencies.find_report(
        target_issue,
        report_dir,
    )
    if error_message:
        raise SystemExit(error_message)

    try:
        single_bets, duplex_bet = dependencies.parse_bets(report_filepath)
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f'错误: 无法解析报告 {report_filepath}: {exc}') from exc
    if not single_bets or not duplex_bet['red'] or not duplex_bet['blue']:
        raise SystemExit(f'错误: 未能从报告 {report_filepath} 中成功解析出投注号码。')

    generated_at = generated_at or dependencies.now()
    report = dependencies.format_report(BonusReportData(
        report_filepath=report_filepath,
        target_issue=target_issue,
        winning_reds=latest_draw['red'],
        winning_blue=latest_draw['blue'],
        single_bets=single_bets,
        duplex_bet=duplex_bet,
        generated_at=generated_at,
    ))
    print('\n' + report)

    timestamp = generated_at.strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(report_dir, f'ssq_bonus_check_{timestamp}.txt')
    try:
        dependencies.write_text(filepath, report)
    except OSError as exc:
        raise SystemExit(f'\n写入核对报告文件失败: {exc}') from exc
    print(f'\n核对报告已成功保存到文件: {filepath}')
    return filepath


def main():
    return run_bonus_check()
