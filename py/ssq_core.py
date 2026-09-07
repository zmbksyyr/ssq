"""Shared domain rules for the Double Color Ball lottery."""

import os
import tempfile
from datetime import date, datetime, timedelta
from math import isfinite
from numbers import Integral, Real
from zoneinfo import ZoneInfo

RED_MIN = 1
RED_MAX = 33
BLUE_MIN = 1
BLUE_MAX = 16
RED_COUNT = 6
DRAW_WEEKDAYS = {1, 3, 6}  # Tuesday, Thursday, Sunday
LOCAL_TIMEZONE = ZoneInfo('Asia/Shanghai')

PRIZE_RULES = {
    (6, 1): 5_000_000,
    (6, 0): 100_000,
    (5, 1): 3_000,
    (5, 0): 200,
    (4, 1): 200,
    (4, 0): 10,
    (3, 1): 10,
    (2, 1): 5,
    (1, 1): 5,
    (0, 1): 5,
}

PRIZE_NAMES = {
    (6, 1): "一等奖",
    (6, 0): "二等奖",
    (5, 1): "三等奖",
    (5, 0): "四等奖",
    (4, 1): "四等奖",
    (4, 0): "五等奖",
    (3, 1): "五等奖",
    (2, 1): "六等奖",
    (1, 1): "六等奖",
    (0, 1): "六等奖",
}


def parse_integer(value, field_name):
    if isinstance(value, bool):
        raise TypeError(f"{field_name}必须为整数")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        if not isfinite(value) or not float(value).is_integer():
            raise ValueError(f"{field_name}必须为整数")
        return int(value)
    text = str(value).strip()
    if not text or not text.lstrip('+-').isdigit():
        raise ValueError(f"{field_name}必须为整数")
    return int(text)


def local_now():
    return datetime.now(LOCAL_TIMEZONE)


def local_today():
    return local_now().date()


def parse_issue(value):
    issue = parse_integer(value, '期号')
    year, sequence = divmod(issue, 1000)
    if year < 2003 or not 1 <= sequence <= 999:
        raise ValueError(f"无效期号: {value}")
    return issue


def parse_red_balls(value, expected_count=RED_COUNT):
    if isinstance(value, str):
        numbers = [parse_integer(part, '红球号码') for part in value.split(',')]
    else:
        numbers = [parse_integer(part, '红球号码') for part in value]
    if len(numbers) != expected_count:
        raise ValueError(f"红球数量必须为 {expected_count}，实际为 {len(numbers)}")
    if len(set(numbers)) != expected_count:
        raise ValueError("红球号码不能重复")
    if any(number < RED_MIN or number > RED_MAX for number in numbers):
        raise ValueError(f"红球号码必须在 {RED_MIN} 到 {RED_MAX} 之间")
    return sorted(numbers)


def parse_blue_ball(value):
    number = parse_integer(value, '蓝球号码')
    if number < BLUE_MIN or number > BLUE_MAX:
        raise ValueError(f"蓝球号码必须在 {BLUE_MIN} 到 {BLUE_MAX} 之间")
    return number


def parse_blue_balls(value):
    if isinstance(value, str):
        numbers = [parse_blue_ball(part.strip()) for part in value.split(',')]
    else:
        numbers = [parse_blue_ball(part) for part in value]
    if not numbers:
        raise ValueError("蓝球号码不能为空")
    if len(set(numbers)) != len(numbers):
        raise ValueError("蓝球号码不能重复")
    return numbers


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
        raise ValueError(f"期号年份 {issue_year} 与开奖日期 {draw_date} 不一致")

    next_draw_date = draw_date + timedelta(days=1)
    while next_draw_date.weekday() not in DRAW_WEEKDAYS:
        next_draw_date += timedelta(days=1)
    if next_draw_date.year != issue_year:
        return next_draw_date.year * 1000 + 1
    return issue + 1


def atomic_write_text(path, content, encoding='utf-8'):
    """Write text via an adjacent temporary file and atomically replace the target."""
    target = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding=encoding, newline='', dir=directory,
            prefix='.ssq-', suffix='.tmp', delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            temporary_file.write(content)
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
