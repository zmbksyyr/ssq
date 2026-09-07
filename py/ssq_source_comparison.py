"""Comparison rules for authoritative and secondary lottery sources."""

import logging

from ssq_parsing import parse_issue

logger = logging.getLogger('ssq_data_processor')


def cross_check_sources(primary_records, secondary_records):
    """Report disagreements while retaining the authoritative TXT values."""
    secondary_by_issue = {str(item['期号']): item for item in secondary_records}
    mismatches = []
    checked = 0
    for item in primary_records:
        other = secondary_by_issue.get(str(item['期号']))
        if other is None:
            continue
        checked += 1
        if item['红球'] != other['红球'] or item['蓝球'] != other['蓝球']:
            mismatches.append(str(item['期号']))
    if mismatches:
        logger.warning(
            f"数据源号码不一致，期号: {', '.join(mismatches)}；保留 TXT 权威数据。"
        )
    elif checked:
        logger.info(f'两个数据源交叉核验通过，共 {checked} 期重叠记录。')
    else:
        logger.warning('两个数据源没有可交叉核验的重叠期号。')
    return mismatches


def find_secondary_only_issues(primary_records, secondary_records):
    """Return issues visible to the cross-check source but absent from TXT."""
    primary_issues = {str(item['期号']) for item in primary_records}
    secondary_issues = {str(item['期号']) for item in secondary_records}
    return sorted(
        secondary_issues - primary_issues,
        key=parse_issue,
    )
