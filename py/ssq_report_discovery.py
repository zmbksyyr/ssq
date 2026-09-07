"""Analysis-report metadata parsing and target-issue discovery."""

import glob
import os
import re

from ssq_parsing import parse_issue

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


def find_matching_report(target_issue, report_dir):
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
