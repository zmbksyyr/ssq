"""Remote draw sources, parsing, and cross-source verification."""

import logging
from datetime import date

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from ssq_core import parse_blue_ball, parse_issue, parse_red_balls
from urllib3.util.retry import Retry

TXT_DATA_URL = 'https://data.17500.cn/ssq_asc.txt'
HTML_DATA_URL = 'https://www.17500.cn/chart/ssq-tjb.html'
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/91.0.4472.124 Safari/537.36'
    ),
}

logger = logging.getLogger('ssq_data_processor')


def create_http_session():
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=('GET',),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def fetch_latest_data_from_html(url=HTML_DATA_URL, session=None):
    """Fetch number-only records used to cross-check the TXT source."""
    logger.info('正在从HTML网页抓取最新双色球数据...')
    session = session or create_http_session()
    try:
        response = session.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.error(f'从HTML网页获取数据失败: {exc}')
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')
    if not table:
        logger.warning('在网页中未能找到数据表格。')
        return []

    records = []
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        if len(cells) < 3:
            continue
        try:
            issue = parse_issue(cells[0].get_text(strip=True).removesuffix('期'))
            red_numbers = parse_red_balls(
                cells[1].get_text(' ', strip=True).split()
            )
            blue_number = parse_blue_ball(cells[2].get_text(strip=True))
            records.append({
                '期号': str(issue),
                '红球': ','.join(f'{number:02d}' for number in red_numbers),
                '蓝球': f'{blue_number:02d}',
            })
        except (TypeError, ValueError, IndexError) as exc:
            logger.warning(
                f'解析表格行时出错: {row.text.strip()}. 错误: {exc}. 跳过此行。'
            )
    logger.info(f'从HTML网页成功获取 {len(records)} 期数据。')
    return records


def fetch_full_data_from_txt(url=TXT_DATA_URL, session=None):
    """Download the authoritative history source, including draw dates."""
    logger.info(f'正在从TXT文件源 ({url}) 下载全量数据...')
    session = session or create_http_session()
    try:
        response = session.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        response.encoding = 'utf-8'
        lines = response.text.strip().splitlines()
        logger.info(f'成功下载 {len(lines)} 行数据。')
        return lines
    except requests.exceptions.RequestException as exc:
        logger.error(f'从TXT文件源下载数据失败: {exc}')
        return []


def parse_txt_data(data_lines):
    """Parse authoritative source lines into normalized CSV field values."""
    if not data_lines:
        return []
    logger.info('正在解析TXT数据...')
    parsed_data = []
    for line in data_lines:
        fields = line.strip().split()
        if len(fields) < 9:
            continue
        try:
            issue = parse_issue(fields[0])
            date_text = fields[1]
            red_numbers = parse_red_balls(','.join(fields[2:8]))
            blue_number = parse_blue_ball(fields[8])
            parsed_date = date.fromisoformat(date_text)
            if parsed_date.isoformat() != date_text:
                raise ValueError('日期必须使用 YYYY-MM-DD 格式')
            parsed_data.append([
                str(issue),
                date_text,
                ','.join(f'{number:02d}' for number in red_numbers),
                f'{blue_number:02d}',
            ])
        except (IndexError, TypeError, ValueError) as exc:
            logger.warning(f'解析TXT行失败: {line}. 错误: {exc}')
    logger.info(f'从TXT数据中成功解析出 {len(parsed_data)} 条有效记录。')
    return parsed_data


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
