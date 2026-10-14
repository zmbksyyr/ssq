"""Merged module: bonus_check (generated from 13 modules)."""

import glob
import hashlib
import os
import re
import tempfile
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import pairwise
from math import comb, isfinite
from numbers import Integral, Real
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.errors import ParserError

# --- from ssq_bet_parsing.py ---
SINGLE_HEADER_PATTERN = re.compile('^【单式推荐 \\((\\d+)组\\)】$', re.MULTILINE)


# --- from ssq_bet_parsing.py ---
DUPLEX_HEADER_PATTERN = re.compile('^【7\\+N 复式推荐 \\(1组\\)】$', re.MULTILINE)


# --- from ssq_bet_parsing.py ---
SINGLE_BET_PATTERN = re.compile('^组合\\s+\\d+:\\s+红球\\s+\\[(.*?)\\]\\s+蓝球\\s+\\[(.*?)\\]$')


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


# --- from ssq_bet_parsing.py ---
def parse_single_bet_line(line):
    match = SINGLE_BET_PATTERN.fullmatch(line)
    if match is None:
        raise ValueError('字段格式不完整')
    return {'red': parse_red_balls(match.group(1)), 'blue': parse_blue_ball(match.group(2))}


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


# --- from ssq_bet_parsing.py ---
def parse_duplex_section(section):
    red_match = re.search('^\\s*红球:\\s*\\[(.*?)\\]\\s*$', section, re.MULTILINE)
    blue_match = re.search('^\\s*蓝球:\\s*\\[(.*?)\\]\\s*$', section, re.MULTILINE)
    if red_match is None or blue_match is None:
        raise ValueError('复式投注内容不完整')
    return {'red': parse_red_balls(red_match.group(1), expected_count=7), 'blue': parse_blue_balls(blue_match.group(1))}


# --- from ssq_bet_parsing.py ---
def validate_parsed_bets(expected_count, single_bets):
    if len(single_bets) != expected_count:
        raise ValueError(f'单式投注数量不完整: 声明 {expected_count} 注，实际解析 {len(single_bets)} 注')
    unique_singles = {(tuple(bet['red']), bet['blue']) for bet in single_bets}
    if len(unique_singles) != len(single_bets):
        raise ValueError('报告包含重复单式投注')


# --- from ssq_bet_parsing.py ---
def parse_report_bets_content(content):
    """Parse a complete set of single and duplex bets from report text."""
    single_header = SINGLE_HEADER_PATTERN.search(content)
    if single_header is None:
        raise ValueError('报告缺少或无法识别单式推荐标题')
    duplex_header = DUPLEX_HEADER_PATTERN.search(content, single_header.end())
    if duplex_header is None:
        raise ValueError('报告缺少或无法识别复式推荐标题')
    single_section = content[single_header.end():duplex_header.start()]
    single_lines = [line.strip() for line in single_section.splitlines() if line.strip().startswith('组合')]
    try:
        single_bets = [parse_single_bet_line(line) for line in single_lines]
    except (TypeError, ValueError) as exc:
        raise ValueError(f'单式投注解析失败: {exc}') from exc
    validate_parsed_bets(int(single_header.group(1)), single_bets)
    try:
        duplex_bet = parse_duplex_section(content[duplex_header.end():])
    except (TypeError, ValueError) as exc:
        raise ValueError(f'复式投注解析失败: {exc}') from exc
    return (single_bets, duplex_bet)


# --- from ssq_bet_parsing.py ---
def parse_report_bets(filepath):
    """Read and parse all single and duplex bets from an analysis report."""
    with open(filepath, encoding='utf-8') as report_file:
        return parse_report_bets_content(report_file.read())


# --- from ssq_bonus_reporting.py ---
@dataclass(frozen=True)
class BonusReportData:
    report_filepath: str | os.PathLike
    target_issue: int
    winning_reds: Collection[int]
    winning_blue: int
    single_bets: Sequence[Mapping]
    duplex_bet: Mapping
    generated_at: datetime


# --- from ssq_domain.py ---
PRIZE_NAMES = {(6, 1): '一等奖', (6, 0): '二等奖', (5, 1): '三等奖', (5, 0): '四等奖', (4, 1): '四等奖', (4, 0): '五等奖', (3, 1): '五等奖', (2, 1): '六等奖', (1, 1): '六等奖', (0, 1): '六等奖'}


# --- from ssq_domain.py ---
PRIZE_RULES = {(6, 1): 5000000, (6, 0): 100000, (5, 1): 3000, (5, 0): 200, (4, 1): 200, (4, 0): 10, (3, 1): 10, (2, 1): 5, (1, 1): 5, (0, 1): 5}


# --- from ssq_prize_calculation.py ---
def calculate_duplex_prize(bet_reds, bet_blues, winning_reds, winning_blue):
    """Calculate every winning subticket contained in a 7+N ticket."""
    bet_reds = parse_red_balls(bet_reds, expected_count=7)
    bet_blues = parse_blue_balls(bet_blues)
    winning_reds = set(parse_red_balls(winning_reds))
    winning_blue = parse_blue_ball(winning_blue)
    total_prize = 0
    prize_breakdown = {}
    red_hits = len(set(bet_reds) & winning_reds)
    red_misses = len(bet_reds) - red_hits
    unique_blues = set(bet_blues)
    blue_hit = int(winning_blue in unique_blues)
    for (red_needed, blue_needed), prize_value in PRIZE_RULES.items():
        if red_needed > red_hits:
            continue
        red_combos = comb(red_hits, red_needed) * comb(red_misses, 6 - red_needed)
        blue_combos = blue_hit if blue_needed else len(unique_blues) - blue_hit
        winning_tickets = red_combos * blue_combos
        if winning_tickets:
            prize_name = PRIZE_NAMES[red_needed, blue_needed]
            prize_breakdown[prize_name] = prize_breakdown.get(prize_name, 0) + winning_tickets
            total_prize += winning_tickets * prize_value
    summary = f'总计命中 {red_hits} 个红球, {blue_hit} 个蓝球'
    return (total_prize, prize_breakdown, summary)


# --- from ssq_prize_calculation.py ---
def calculate_single_prize(bet_reds, bet_blue, winning_reds, winning_blue):
    """Calculate the fixed-reference prize for one standard ticket."""
    bet_red_set = set(parse_red_balls(bet_reds))
    winning_red_set = set(parse_red_balls(winning_reds))
    bet_blue = parse_blue_ball(bet_blue)
    winning_blue = parse_blue_ball(winning_blue)
    red_hits = len(bet_red_set & winning_red_set)
    blue_hit = int(bet_blue == winning_blue)
    hit_key = (red_hits, blue_hit)
    return (PRIZE_RULES.get(hit_key, 0), PRIZE_NAMES.get(hit_key, '未中奖'), f'命中{red_hits}+{blue_hit}')


# --- from ssq_bonus_reporting.py ---
def format_bonus_report(data):
    """Build one deterministic report from validated draws and bets."""
    if not isinstance(data, BonusReportData):
        raise TypeError('data 必须为 BonusReportData')
    total_single_bonus = 0
    single_details = []
    for index, bet in enumerate(data.single_bets, 1):
        prize, prize_name, summary = calculate_single_prize(bet['red'], bet['blue'], data.winning_reds, data.winning_blue)
        total_single_bonus += prize
        single_details.append(f"  组合 {index:>2}: {bet['red']!s:<24} 蓝球 [{bet['blue']:02d}] -> {summary}, {prize_name}, 参考奖金: {prize} 元")
    duplex_prize, duplex_breakdown, duplex_summary = calculate_duplex_prize(data.duplex_bet['red'], data.duplex_bet['blue'], data.winning_reds, data.winning_blue)
    lines = ['=' * 70, '          双色球推荐核对报告', '=' * 70, f"\n报告生成时间: {data.generated_at.strftime('%Y-%m-%d %H:%M:%S')}", f'核对报告文件: {os.path.basename(data.report_filepath)}', f'核对开奖期数: {data.target_issue}', f'官方开奖号码: 红球 {sorted(data.winning_reds)}  蓝球 [{data.winning_blue}]', '奖金说明: 使用固定参考金额估算；一等奖、二等奖实际金额以官方派奖为准。', '\n--- 1. 单式推荐核对详情 ---', *single_details, f'\n单式推荐参考奖金: {total_single_bonus} 元', '\n--- 2. 复式推荐核对详情 ---', f"  红球: {data.duplex_bet['red']}", f"  蓝球: {data.duplex_bet['blue']}", f'  核对结果: {duplex_summary}']
    if not duplex_breakdown:
        lines.append('  奖项构成: 未中任何奖项。')
    else:
        prize_order = {name: index for index, name in enumerate(PRIZE_NAMES.values())}
        lines.append('  奖项构成:')
        for name, count in sorted(duplex_breakdown.items(), key=lambda item: prize_order[item[0]]):
            lines.append(f'    - {name}: {count} 注')
    lines.extend([f'\n复式推荐参考奖金: {duplex_prize} 元', '\n' + '-' * 70, f'总计参考奖金: {total_single_bonus + duplex_prize} 元', '=' * 70])
    return '\n'.join(lines)


# --- from ssq_bonus_reporting.py ---
def build_bonus_report(report_filepath, target_issue, winning_reds, winning_blue, single_bets, duplex_bet, generated_at):
    """Compatibility wrapper for data-object based report formatting."""
    return format_bonus_report(BonusReportData(report_filepath=report_filepath, target_issue=target_issue, winning_reds=winning_reds, winning_blue=winning_blue, single_bets=single_bets, duplex_bet=duplex_bet, generated_at=generated_at))


# --- from ssq_bonus_workflow.py ---
@dataclass(frozen=True)
class BonusCheckDependencies:
    load_draw: Callable[..., Any]
    find_report: Callable[..., Any]
    parse_bets: Callable[..., Any]
    format_report: Callable[..., str]
    write_text: Callable[..., Any]
    now: Callable[..., Any]


# --- from ssq_parsing.py ---
def parse_issue(value):
    issue = parse_integer(value, '期号')
    year, sequence = divmod(issue, 1000)
    if year < 2003 or not 1 <= sequence <= 999:
        raise ValueError(f'无效期号: {value}')
    return issue


# --- from ssq_report_discovery.py ---
TARGET_ISSUE_PATTERN = re.compile('^Prediction_Target_Issue:[ \\t]*(\\S+)[ \\t]*$', re.MULTILINE)


# --- from ssq_report_discovery.py ---
def parse_report_target_issue(content):
    """Return the report's single validated prediction target issue."""
    matches = TARGET_ISSUE_PATTERN.findall(content)
    if len(matches) != 1:
        raise ValueError(f'报告目标期号元数据必须恰好出现一次，实际 {len(matches)} 次')
    return parse_issue(matches[0])


# --- from ssq_report_discovery.py ---
def find_matching_report(target_issue, report_dir):
    """Return the newest analysis report targeting the requested issue."""
    target_issue = parse_issue(target_issue)
    report_pattern = os.path.join(report_dir, 'ssq_analysis_output_*.txt')
    report_files = glob.glob(report_pattern)
    if not report_files:
        return (None, "错误: 当前目录未找到任何 'ssq_analysis_output_*.txt' 报告文件。")
    for report_file in sorted(report_files, reverse=True):
        try:
            with open(report_file, encoding='utf-8') as handle:
                prediction_target = parse_report_target_issue(handle.read())
            if prediction_target == target_issue:
                return (report_file, None)
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            print(f'警告: 读取文件 {report_file} 时出错: {exc}')
    return (None, f"错误: 未找到预测目标期号为 '{target_issue}' 的报告文件。")


# --- from ssq_domain.py ---
DRAW_COLUMNS = ('期号', '日期', '红球', '蓝球')


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


# --- from ssq_latest_draw.py ---
def load_latest_draw(filepath, normalize=None, validate_dates=None):
    """Return the latest draw after applying the shared history contract."""
    normalize = normalize if normalize is not None else normalize_draw_frame
    validate_dates = validate_dates if validate_dates is not None else validate_draw_dates_not_future
    frame = normalize(pd.read_csv(filepath, header=0))
    validate_dates(frame)
    if frame.empty:
        raise ValueError('开奖数据为空')
    latest = frame.iloc[-1]
    return {'issue': int(latest['期号']), 'red': set(latest['红球']), 'blue': latest['蓝球']}


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


# --- from ssq_file_io.py ---
def atomic_write_text(path, content, encoding='utf-8'):
    """Atomically replace a text file with the supplied content."""
    with atomic_text_writer(path, encoding=encoding) as temporary_file:
        temporary_file.write(content)


# --- from ssq_bonus_workflow.py ---
def default_bonus_dependencies():
    return BonusCheckDependencies(load_draw=load_latest_draw, find_report=find_matching_report, parse_bets=parse_report_bets, format_report=format_bonus_report, write_text=atomic_write_text, now=local_now)


# --- from ssq_paths.py ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# --- from ssq_paths.py ---
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# --- from ssq_paths.py ---
CSV_PATH = os.path.join(PROJECT_ROOT, 'shuangseqiu.csv')


# --- from ssq_paths.py ---
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')


# --- from ssq_bonus_workflow.py ---
def run_bonus_check(csv_path=CSV_PATH, report_dir=REPORT_DIR, generated_at=None, dependencies=None):
    """Run a complete prize check and return the generated report path."""
    dependencies = dependencies or default_bonus_dependencies()
    try:
        latest_draw = dependencies.load_draw(csv_path)
    except (OSError, UnicodeError, TypeError, ValueError, ParserError) as exc:
        raise SystemExit(f'读取 {csv_path} 文件失败: {exc}') from exc
    target_issue = latest_draw['issue']
    report_filepath, error_message = dependencies.find_report(target_issue, report_dir)
    if error_message:
        raise SystemExit(error_message)
    try:
        single_bets, duplex_bet = dependencies.parse_bets(report_filepath)
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f'错误: 无法解析报告 {report_filepath}: {exc}') from exc
    if not single_bets or not duplex_bet['red'] or (not duplex_bet['blue']):
        raise SystemExit(f'错误: 未能从报告 {report_filepath} 中成功解析出投注号码。')
    generated_at = generated_at or dependencies.now()
    report = dependencies.format_report(BonusReportData(report_filepath=report_filepath, target_issue=target_issue, winning_reds=latest_draw['red'], winning_blue=latest_draw['blue'], single_bets=single_bets, duplex_bet=duplex_bet, generated_at=generated_at))
    print('\n' + report)
    timestamp = generated_at.strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(report_dir, f'ssq_bonus_check_{timestamp}.txt')
    try:
        dependencies.write_text(filepath, report)
    except OSError as exc:
        raise SystemExit(f'\n写入核对报告文件失败: {exc}') from exc
    print(f'\n核对报告已成功保存到文件: {filepath}')
    return filepath


# --- from ssq_bonus_workflow.py ---
def main():
    return run_bonus_check()


# --- from ssq_domain.py ---
RED_BALLS = tuple(range(RED_MIN, RED_MAX + 1))


# --- from ssq_domain.py ---
BLUE_BALLS = tuple(range(BLUE_MIN, BLUE_MAX + 1))


# --- from ssq_domain.py ---
PRIME_RED_BALLS = frozenset({2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31})


# --- from ssq_draw_data.py ---
def serialize_draw_frame(frame):
    """Return a validated draw table using the stable on-disk CSV format."""
    serialized = normalize_draw_frame(frame)
    serialized['红球'] = serialized['红球'].apply(lambda numbers: ','.join(f'{number:02d}' for number in numbers))
    serialized['蓝球'] = serialized['蓝球'].apply(lambda number: f'{number:02d}')
    return serialized


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


# --- entry point: ssq_bonus_calculation.py ---
if __name__ == '__main__':
    main()
