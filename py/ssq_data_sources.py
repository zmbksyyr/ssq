"""Remote draw sources, parsing, and cross-source verification."""

import logging

import requests
import ssq_http as _http
import ssq_source_comparison as _comparison
import ssq_source_parsing as _source_parsing

TXT_DATA_URL = 'https://data.17500.cn/ssq_asc.txt'
HTML_DATA_URL = 'https://www.17500.cn/chart/ssq-tjb.html'
REQUEST_HEADERS = _http.REQUEST_HEADERS
REQUEST_TIMEOUT_SECONDS = _http.REQUEST_TIMEOUT_SECONDS
create_http_session = _http.create_http_session
fetch_text = _http.fetch_text
parse_html_data = _source_parsing.parse_html_data
parse_txt_data = _source_parsing.parse_txt_data
cross_check_sources = _comparison.cross_check_sources
find_secondary_only_issues = _comparison.find_secondary_only_issues

logger = logging.getLogger('ssq_data_processor')


def fetch_latest_data_from_html(url=HTML_DATA_URL, session=None):
    """Fetch number-only records used to cross-check the TXT source."""
    logger.info('正在从HTML网页抓取最新双色球数据...')
    try:
        content = fetch_text(url, session=session)
    except requests.exceptions.RequestException as exc:
        logger.error(f'从HTML网页获取数据失败: {exc}')
        return []

    records = parse_html_data(content)
    logger.info(f'从HTML网页成功获取 {len(records)} 期数据。')
    return records


def fetch_full_data_from_txt(url=TXT_DATA_URL, session=None):
    """Download the authoritative history source, including draw dates."""
    logger.info(f'正在从TXT文件源 ({url}) 下载全量数据...')
    try:
        content = fetch_text(url, session=session, encoding='utf-8')
        lines = content.strip().splitlines()
        logger.info(f'成功下载 {len(lines)} 行数据。')
        return lines
    except requests.exceptions.RequestException as exc:
        logger.error(f'从TXT文件源下载数据失败: {exc}')
        return []
