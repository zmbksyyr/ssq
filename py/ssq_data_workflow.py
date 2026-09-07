"""Application workflow for updating authoritative draw history."""

import logging
import os
import sys

from ssq_data_sources import (
    HTML_DATA_URL,
    TXT_DATA_URL,
    create_http_session,
    cross_check_sources,
    fetch_full_data_from_txt,
    fetch_latest_data_from_html,
    find_secondary_only_issues,
    parse_txt_data,
)
from ssq_data_store import update_csv_file
from ssq_domain import DRAW_COLUMNS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CSV_FILE_PATH = os.path.join(PROJECT_ROOT, 'shuangseqiu.csv')
logger = logging.getLogger('ssq_data_processor')


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def run_data_update(
    csv_path=CSV_FILE_PATH,
    txt_url=TXT_DATA_URL,
    html_url=HTML_DATA_URL,
    session=None,
):
    """Fetch, cross-check, and persist one authoritative full snapshot."""
    logger.info('--- 开始执行双色球数据处理任务 ---')
    session = session or create_http_session()
    parsed_rows = parse_txt_data(
        fetch_full_data_from_txt(txt_url, session=session)
    )
    authoritative_records = [
        dict(zip(DRAW_COLUMNS, row, strict=True)) for row in parsed_rows
    ]
    if not authoritative_records:
        raise SystemExit(
            'TXT 权威数据源未返回有效记录，拒绝使用无日期的 HTML 数据更新 CSV。'
        )

    comparison_records = fetch_latest_data_from_html(
        html_url, session=session
    )
    if comparison_records:
        cross_check_sources(authoritative_records, comparison_records)
        missing_authoritative_issues = find_secondary_only_issues(
            authoritative_records,
            comparison_records,
        )
        if missing_authoritative_issues:
            raise SystemExit(
                'HTML 交叉源包含 TXT 权威源缺失的期号: '
                f"{', '.join(missing_authoritative_issues)}；拒绝更新 CSV。"
            )

    if not update_csv_file(
        csv_path,
        authoritative_records,
        require_full_snapshot=True,
    ):
        raise SystemExit('权威开奖数据更新失败。')
    logger.info('--- 双色球数据处理任务完成 ---')
    return os.fspath(csv_path)


def main():
    configure_logging()
    return run_data_update()
