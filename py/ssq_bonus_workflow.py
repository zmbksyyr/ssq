"""Application workflow for checking recommendations against a draw."""

import glob
import os
import re

import pandas as pd
from ssq_bonus_reporting import build_bonus_report
from ssq_core import atomic_write_text, local_now, parse_issue
from ssq_draw_data import normalize_draw_frame
from ssq_prizes import parse_report_bets

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(PROJECT_ROOT, 'shuangseqiu.csv')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')
TARGET_ISSUE_PATTERN = re.compile(
    r'^Prediction_Target_Issue:[ \t]*(\S+)[ \t]*$',
    re.MULTILINE,
)


def parse_report_target_issue(content):
    """Return the report's single validated prediction target issue."""
    matches = TARGET_ISSUE_PATTERN.findall(content)
    if len(matches) != 1:
        raise ValueError(f'报告目标期号元数据必须恰好出现一次，实际 {len(matches)} 次')
    return parse_issue(matches[0])


def find_matching_report(target_issue, report_dir=REPORT_DIR):
    """Return the newest analysis report targeting the requested issue."""
    target_issue = parse_issue(target_issue)
    report_pattern = os.path.join(report_dir, 'ssq_analysis_output_*.txt')
    report_files = glob.glob(report_pattern)
    if not report_files:
        return None, "错误: 当前目录未找到任何 'ssq_analysis_output_*.txt' 报告文件。"

    for report_file in sorted(report_files, reverse=True):
        try:
            with open(report_file, encoding='utf-8') as handle:
                prediction_target = parse_report_target_issue(handle.read())
            if prediction_target == target_issue:
                return report_file, None
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            print(f'警告: 读取文件 {report_file} 时出错: {exc}')
    return None, f"错误: 未找到预测目标期号为 '{target_issue}' 的报告文件。"


def load_latest_draw(filepath=CSV_PATH):
    frame = normalize_draw_frame(pd.read_csv(filepath, header=0))
    if frame.empty:
        raise ValueError('开奖数据为空')
    latest = frame.iloc[-1]
    return {
        'issue': int(latest['期号']),
        'red': set(latest['红球']),
        'blue': latest['蓝球'],
    }


def run_bonus_check(
    csv_path=CSV_PATH,
    report_dir=REPORT_DIR,
    generated_at=None,
):
    """Run a complete prize check and return the generated report path."""
    try:
        latest_draw = load_latest_draw(csv_path)
    except (
        OSError, UnicodeError, TypeError, ValueError, pd.errors.ParserError
    ) as exc:
        raise SystemExit(f'读取 {csv_path} 文件失败: {exc}') from exc

    target_issue = latest_draw['issue']
    report_filepath, error_message = find_matching_report(target_issue, report_dir)
    if error_message:
        raise SystemExit(error_message)

    try:
        single_bets, duplex_bet = parse_report_bets(report_filepath)
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f'错误: 无法解析报告 {report_filepath}: {exc}') from exc
    if not single_bets or not duplex_bet['red'] or not duplex_bet['blue']:
        raise SystemExit(f'错误: 未能从报告 {report_filepath} 中成功解析出投注号码。')

    generated_at = generated_at or local_now()
    report = build_bonus_report(
        report_filepath,
        target_issue,
        latest_draw['red'],
        latest_draw['blue'],
        single_bets,
        duplex_bet,
        generated_at,
    )
    print('\n' + report)

    timestamp = generated_at.strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(report_dir, f'ssq_bonus_check_{timestamp}.txt')
    try:
        atomic_write_text(filepath, report)
    except OSError as exc:
        raise SystemExit(f'\n写入核对报告文件失败: {exc}') from exc
    print(f'\n核对报告已成功保存到文件: {filepath}')
    return filepath


def main():
    return run_bonus_check()
