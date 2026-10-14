"""Merged module: data_prep (generated from 15 modules)."""

import csv
import hashlib
import logging
import os
import sys
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import pairwise
from math import isfinite
from numbers import Integral, Real
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- from ssq_csv_store.py ---
logger = logging.getLogger('ssq_data_processor')


# --- from ssq_csv_store.py ---
def read_existing_csv(csv_path):
    """Read an existing draw CSV, returning an empty frame when absent."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        logger.info('CSV file does not exist or is empty; a new file will be created.')
        return pd.DataFrame()
    logger.info('Reading existing CSV file: %s', csv_path)
    try:
        return pd.read_csv(csv_path, dtype={'期号': str}, encoding='utf-8')
    except (UnicodeDecodeError, pd.errors.ParserError):
        logger.warning('Unable to read CSV as UTF-8; trying GBK.')
        return pd.read_csv(csv_path, dtype={'期号': str}, encoding='gbk')
    except pd.errors.EmptyDataError:
        logger.warning('Existing CSV file is empty.')
        return pd.DataFrame()


# --- from ssq_file_io.py ---
@contextmanager
def atomic_text_writer(path, encoding='utf-8'):
    """Yield an adjacent temporary text file and replace the target on success."""
    target = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding=encoding, newline='', dir=directory, prefix='.ssq-', suffix='.tmp', delete=False) as temporary_file:
            temporary_path = temporary_file.name
            yield temporary_file
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


# --- from ssq_csv_store.py ---
def atomic_write_csv(frame, csv_path):
    """Atomically replace a CSV with the supplied frame."""
    with atomic_text_writer(csv_path) as temporary_file:
        frame.to_csv(temporary_file, index=False, quoting=csv.QUOTE_MINIMAL)


# --- from ssq_data_sources.py ---
TXT_DATA_URL = 'https://data.17500.cn/ssq_asc.txt'


# --- from ssq_data_sources.py ---
HTML_DATA_URL = 'https://www.17500.cn/chart/ssq-tjb.html'


# --- from ssq_http.py ---
REQUEST_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}


# --- from ssq_http.py ---
REQUEST_TIMEOUT_SECONDS = 30


# --- from ssq_http.py ---
def create_http_session():
    retry = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=('GET',), respect_retry_after_header=True)
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


# --- from ssq_http.py ---
def fetch_text(url, session=None, encoding=None):
    """Fetch one text resource using the shared request policy."""
    session = session or create_http_session()
    response = session.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    if encoding is not None:
        response.encoding = encoding
    return response.text


# --- from ssq_domain.py ---
BLUE_MAX = 16


# --- from ssq_domain.py ---
BLUE_MIN = 1


# --- from ssq_parsing.py ---
def parse_integer(value, field_name):
    if isinstance(value, bool):
        raise TypeError(f'{field_name}必须为整数')
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        if not isfinite(value) or not float(value).is_integer():
            raise ValueError(f'{field_name}必须为整数')
        return int(value)
    text = str(value).strip()
    if not text or not text.lstrip('+-').isdigit():
        raise ValueError(f'{field_name}必须为整数')
    return int(text)


# --- from ssq_parsing.py ---
def parse_blue_ball(value):
    number = parse_integer(value, '蓝球号码')
    if number < BLUE_MIN or number > BLUE_MAX:
        raise ValueError(f'蓝球号码必须在 {BLUE_MIN} 到 {BLUE_MAX} 之间')
    return number


# --- from ssq_parsing.py ---
def parse_issue(value):
    issue = parse_integer(value, '期号')
    year, sequence = divmod(issue, 1000)
    if year < 2003 or not 1 <= sequence <= 999:
        raise ValueError(f'无效期号: {value}')
    return issue


# --- from ssq_domain.py ---
RED_COUNT = 6


# --- from ssq_domain.py ---
RED_MAX = 33


# --- from ssq_domain.py ---
RED_MIN = 1


# --- from ssq_parsing.py ---
def parse_red_balls(value, expected_count=RED_COUNT):
    if isinstance(value, str):
        numbers = [parse_integer(part, '红球号码') for part in value.split(',')]
    else:
        numbers = [parse_integer(part, '红球号码') for part in value]
    if len(numbers) != expected_count:
        raise ValueError(f'红球数量必须为 {expected_count}，实际为 {len(numbers)}')
    if len(set(numbers)) != expected_count:
        raise ValueError('红球号码不能重复')
    if any(number < RED_MIN or number > RED_MAX for number in numbers):
        raise ValueError(f'红球号码必须在 {RED_MIN} 到 {RED_MAX} 之间')
    return sorted(numbers)


# --- from ssq_source_parsing.py ---
def parse_html_data(content):
    """Parse number-only comparison records from the source HTML."""
    soup = BeautifulSoup(content, 'html.parser')
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
            red_numbers = parse_red_balls(cells[1].get_text(' ', strip=True).split())
            blue_number = parse_blue_ball(cells[2].get_text(strip=True))
            records.append({'期号': str(issue), '红球': ','.join(f'{number:02d}' for number in red_numbers), '蓝球': f'{blue_number:02d}'})
        except (TypeError, ValueError, IndexError) as exc:
            logger.warning(f'解析表格行时出错: {row.text.strip()}. 错误: {exc}. 跳过此行。')
    return records


# --- from ssq_data_sources.py ---
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


# --- from ssq_data_sources.py ---
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


# --- from ssq_data_workflow.py ---
def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])


# --- from ssq_paths.py ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# --- from ssq_paths.py ---
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# --- from ssq_paths.py ---
CSV_PATH = os.path.join(PROJECT_ROOT, 'shuangseqiu.csv')


# --- from ssq_domain.py ---
DRAW_COLUMNS = ('期号', '日期', '红球', '蓝球')


# --- from ssq_source_comparison.py ---
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
        logger.warning(f"数据源号码不一致，期号: {', '.join(mismatches)}；保留 TXT 权威数据。")
    elif checked:
        logger.info(f'两个数据源交叉核验通过，共 {checked} 期重叠记录。')
    else:
        logger.warning('两个数据源没有可交叉核验的重叠期号。')
    return mismatches


# --- from ssq_source_comparison.py ---
def find_secondary_only_issues(primary_records, secondary_records):
    """Return issues visible to the cross-check source but absent from TXT."""
    primary_issues = {str(item['期号']) for item in primary_records}
    secondary_issues = {str(item['期号']) for item in secondary_records}
    return sorted(secondary_issues - primary_issues, key=parse_issue)


# --- from ssq_source_parsing.py ---
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
            parsed_data.append([str(issue), date_text, ','.join(f'{number:02d}' for number in red_numbers), f'{blue_number:02d}'])
        except (IndexError, TypeError, ValueError) as exc:
            logger.warning(f'解析TXT行失败: {line}. 错误: {exc}')
    logger.info(f'从TXT数据中成功解析出 {len(parsed_data)} 条有效记录。')
    return parsed_data


# --- from ssq_domain.py ---
DRAW_WEEKDAYS = {1, 3, 6}


# --- from ssq_draw_data.py ---
def normalize_draw_frame(frame):
    """Return a validated draw table in chronological, typed form."""
    actual_columns = set(frame.columns)
    expected_columns = set(DRAW_COLUMNS)
    if actual_columns != expected_columns or len(frame.columns) != len(DRAW_COLUMNS):
        missing = sorted(expected_columns - actual_columns)
        unexpected = sorted(actual_columns - expected_columns)
        raise ValueError(f'字段不匹配: 缺少 {missing}, 多余 {unexpected}')
    normalized = frame.loc[:, list(DRAW_COLUMNS)].copy()
    normalized['期号'] = normalized['期号'].apply(parse_issue)
    if normalized['期号'].duplicated().any():
        duplicates = sorted(int(issue) for issue in normalized.loc[normalized['期号'].duplicated(keep=False), '期号'].unique())
        raise ValueError(f'存在重复期号: {duplicates}')
    parsed_dates = pd.to_datetime(normalized['日期'], format='%Y-%m-%d', errors='raise')
    normalized['日期'] = parsed_dates
    normalized['红球'] = normalized['红球'].apply(parse_red_balls)
    normalized['蓝球'] = normalized['蓝球'].apply(parse_blue_ball)
    normalized = normalized.sort_values('期号').reset_index(drop=True)
    if (normalized['期号'] // 1000 != normalized['日期'].dt.year).any():
        raise ValueError('期号年份与开奖日期不一致')
    if normalized['日期'].duplicated().any():
        raise ValueError('同一开奖日期不能对应多个期号')
    if not normalized['日期'].is_monotonic_increasing:
        raise ValueError('期号与开奖日期顺序不一致')
    if not normalized['日期'].dt.weekday.isin(DRAW_WEEKDAYS).all():
        raise ValueError('开奖日期必须为周二、周四或周日')
    issues = normalized['期号'].tolist()
    for previous, current in pairwise(issues):
        previous_year, previous_sequence = divmod(previous, 1000)
        current_year, current_sequence = divmod(current, 1000)
        continuous = current_year == previous_year and current_sequence == previous_sequence + 1 or (current_year == previous_year + 1 and current_sequence == 1)
        if not continuous:
            raise ValueError(f'开奖期号不连续: {previous} -> {current}')
    normalized['日期'] = normalized['日期'].dt.strftime('%Y-%m-%d')
    return normalized


# --- from ssq_draw_data.py ---
def serialize_draw_frame(frame):
    """Return a validated draw table using the stable on-disk CSV format."""
    serialized = normalize_draw_frame(frame)
    serialized['红球'] = serialized['红球'].apply(lambda numbers: ','.join(f'{number:02d}' for number in numbers))
    serialized['蓝球'] = serialized['蓝球'].apply(lambda number: f'{number:02d}')
    return serialized


# --- from ssq_snapshot_validation.py ---
def normalize_lottery_frame(frame):
    """Validate and normalize a draw DataFrame before it reaches the CSV."""
    return serialize_draw_frame(frame)


# --- from ssq_snapshot_validation.py ---
MIN_FULL_SNAPSHOT_RECORDS = 100


# --- from ssq_domain.py ---
LOCAL_TIMEZONE = ZoneInfo('Asia/Shanghai')


# --- from ssq_draw_schedule.py ---
def local_now():
    return datetime.now(LOCAL_TIMEZONE)


# --- from ssq_draw_schedule.py ---
def local_today():
    return local_now().date()


# --- from ssq_draw_data.py ---
def validate_draw_dates_not_future(frame, today=None):
    """Reject draw records dated after the current local calendar date."""
    if frame.empty:
        return
    today = today or local_today()
    latest_date = pd.to_datetime(frame['日期'], format='%Y-%m-%d', errors='raise').max().date()
    if latest_date > today:
        raise ValueError(f'开奖记录包含未来开奖日期: {latest_date}')


# --- from ssq_snapshot_validation.py ---
def validate_authoritative_snapshot(new_data, existing_data, today=None):
    """Ensure the advertised full snapshot cannot truncate local history."""
    if len(new_data) < MIN_FULL_SNAPSHOT_RECORDS:
        raise ValueError(f'Authoritative snapshot has only {len(new_data)} records; at least {MIN_FULL_SNAPSHOT_RECORDS} are required.')
    validate_draw_dates_not_future(new_data, today=today)
    if existing_data.empty:
        return
    missing_issues = sorted(set(existing_data['期号']) - set(new_data['期号']))
    if missing_issues:
        preview = ', '.join(str(issue) for issue in missing_issues[:5])
        suffix = ' ...' if len(missing_issues) > 5 else ''
        raise ValueError(f'Authoritative snapshot is missing local issues: {preview}{suffix}')


# --- from ssq_history_store.py ---
@dataclass(frozen=True)
class HistoryStoreDependencies:
    """Replaceable boundaries used by the draw-history merge workflow."""
    normalize_frame: Callable[[Any], pd.DataFrame] = normalize_lottery_frame
    read_csv: Callable[[Any], pd.DataFrame] = read_existing_csv
    validate_snapshot: Callable[..., None] = validate_authoritative_snapshot
    write_csv: Callable[[pd.DataFrame, Any], None] = atomic_write_csv


# --- from ssq_history_store.py ---
def update_csv_file(csv_path, all_new_data, require_full_snapshot=False, dependencies=None):
    """Validate and merge new records, preferring new values by issue."""
    if not all_new_data:
        logger.info('No new records; leaving the CSV unchanged.')
        return False
    dependencies = dependencies or HistoryStoreDependencies()
    try:
        new_data = dependencies.normalize_frame(pd.DataFrame(all_new_data))
        existing_data = dependencies.read_csv(csv_path)
        if not existing_data.empty:
            existing_data = dependencies.normalize_frame(existing_data)
        if require_full_snapshot:
            dependencies.validate_snapshot(new_data, existing_data)
        combined = pd.concat([existing_data, new_data], ignore_index=True) if not existing_data.empty else new_data
        final_data = dependencies.normalize_frame(combined.drop_duplicates(subset=['期号'], keep='last'))
        dependencies.write_csv(final_data, csv_path)
        logger.info('CSV file updated at %s with %s records.', csv_path, len(final_data))
        return True
    except (OSError, UnicodeError, TypeError, ValueError, pd.errors.ParserError) as exc:
        logger.error('Unable to update CSV file: %s', exc)
        return False


# --- from ssq_data_workflow.py ---
def run_data_update(csv_path=CSV_PATH, txt_url=TXT_DATA_URL, html_url=HTML_DATA_URL, session=None):
    """Fetch, cross-check, and persist one authoritative full snapshot."""
    logger.info('--- 开始执行双色球数据处理任务 ---')
    session = session or create_http_session()
    parsed_rows = parse_txt_data(fetch_full_data_from_txt(txt_url, session=session))
    authoritative_records = [dict(zip(DRAW_COLUMNS, row, strict=True)) for row in parsed_rows]
    if not authoritative_records:
        raise SystemExit('TXT 权威数据源未返回有效记录，拒绝使用无日期的 HTML 数据更新 CSV。')
    comparison_records = fetch_latest_data_from_html(html_url, session=session)
    if comparison_records:
        cross_check_sources(authoritative_records, comparison_records)
        missing_authoritative_issues = find_secondary_only_issues(authoritative_records, comparison_records)
        if missing_authoritative_issues:
            raise SystemExit(f"HTML 交叉源包含 TXT 权威源缺失的期号: {', '.join(missing_authoritative_issues)}；拒绝更新 CSV。")
    if not update_csv_file(csv_path, authoritative_records, require_full_snapshot=True):
        raise SystemExit('权威开奖数据更新失败。')
    logger.info('--- 双色球数据处理任务完成 ---')
    return os.fspath(csv_path)


# --- from ssq_data_workflow.py ---
def main():
    configure_logging()
    return run_data_update()


# --- from ssq_domain.py ---
RED_BALLS = tuple(range(RED_MIN, RED_MAX + 1))


# --- from ssq_domain.py ---
BLUE_BALLS = tuple(range(BLUE_MIN, BLUE_MAX + 1))


# --- from ssq_domain.py ---
PRIME_RED_BALLS = frozenset({2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31})


# --- from ssq_domain.py ---
PRIZE_RULES = {(6, 1): 5000000, (6, 0): 100000, (5, 1): 3000, (5, 0): 200, (4, 1): 200, (4, 0): 10, (3, 1): 10, (2, 1): 5, (1, 1): 5, (0, 1): 5}


# --- from ssq_domain.py ---
PRIZE_NAMES = {(6, 1): '一等奖', (6, 0): '二等奖', (5, 1): '三等奖', (5, 0): '四等奖', (4, 1): '四等奖', (4, 0): '五等奖', (3, 1): '五等奖', (2, 1): '六等奖', (1, 1): '六等奖', (0, 1): '六等奖'}


# --- from ssq_draw_data.py ---
def fingerprint_draw_frame(frame):
    """Return a stable fingerprint of normalized, serialized draw history."""
    canonical_csv = serialize_draw_frame(frame).to_csv(index=False, lineterminator='\n')
    return hashlib.sha256(canonical_csv.encode('utf-8')).hexdigest()


# --- from ssq_draw_schedule.py ---
def infer_next_issue(current_issue, current_draw_date):
    """Infer the next regular draw issue, including the year boundary."""
    issue = parse_issue(current_issue)
    issue_year, _ = divmod(issue, 1000)
    if isinstance(current_draw_date, datetime):
        draw_date = current_draw_date.date()
    elif isinstance(current_draw_date, date):
        draw_date = current_draw_date
    else:
        draw_date = date.fromisoformat(str(current_draw_date))
    if draw_date.year != issue_year:
        raise ValueError(f'期号年份 {issue_year} 与开奖日期 {draw_date} 不一致')
    next_draw_date = draw_date + timedelta(days=1)
    while next_draw_date.weekday() not in DRAW_WEEKDAYS:
        next_draw_date += timedelta(days=1)
    if next_draw_date.year != issue_year:
        return next_draw_date.year * 1000 + 1
    return issue + 1


# --- from ssq_file_io.py ---
def atomic_write_text(path, content, encoding='utf-8'):
    """Atomically replace a text file with the supplied content."""
    with atomic_text_writer(path, encoding=encoding) as temporary_file:
        temporary_file.write(content)


# --- from ssq_parsing.py ---
def parse_blue_balls(value):
    if isinstance(value, str):
        numbers = [parse_blue_ball(part.strip()) for part in value.split(',')]
    else:
        numbers = [parse_blue_ball(part) for part in value]
    if not numbers:
        raise ValueError('蓝球号码不能为空')
    if len(set(numbers)) != len(numbers):
        raise ValueError('蓝球号码不能重复')
    return numbers


# --- from ssq_parsing.py ---
def validate_ball_scores(scores, candidates, label):
    """Return numeric scores after enforcing a complete ball domain."""
    expected = set(candidates)
    missing = sorted(expected - set(scores))
    extra = sorted(set(scores) - expected)
    if missing or extra:
        raise ValueError(f'{label}评分键不完整: 缺失 {missing}, 多余 {extra}')
    normalized = {}
    for ball in candidates:
        value = scores[ball]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f'{label} {ball} 的评分必须为数值')
        value = float(value)
        if not isfinite(value):
            raise ValueError(f'{label} {ball} 的评分必须为有限数值')
        normalized[ball] = value
    return normalized


# --- from ssq_paths.py ---
PARAMS_JSON_PATH = os.path.join(PROJECT_ROOT, 'best_params.json')


# --- from ssq_paths.py ---
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')


# --- entry point: ssq_data_processor.py ---
if __name__ == '__main__':
    main()
