"""Shared domain rules for the Double Color Ball lottery."""

import os
import tempfile
from datetime import date, datetime, timedelta

import ssq_domain as _domain
import ssq_parsing as _parsing

RED_MIN = _domain.RED_MIN
RED_MAX = _domain.RED_MAX
BLUE_MIN = _domain.BLUE_MIN
BLUE_MAX = _domain.BLUE_MAX
RED_COUNT = _domain.RED_COUNT
RED_BALLS = _domain.RED_BALLS
BLUE_BALLS = _domain.BLUE_BALLS
PRIME_RED_BALLS = _domain.PRIME_RED_BALLS
DRAW_COLUMNS = _domain.DRAW_COLUMNS
DRAW_WEEKDAYS = _domain.DRAW_WEEKDAYS
LOCAL_TIMEZONE = _domain.LOCAL_TIMEZONE
PRIZE_RULES = _domain.PRIZE_RULES
PRIZE_NAMES = _domain.PRIZE_NAMES


parse_integer = _parsing.parse_integer


def local_now():
    return datetime.now(LOCAL_TIMEZONE)


def local_today():
    return local_now().date()


parse_issue = _parsing.parse_issue
parse_red_balls = _parsing.parse_red_balls
parse_blue_ball = _parsing.parse_blue_ball
parse_blue_balls = _parsing.parse_blue_balls
validate_ball_scores = _parsing.validate_ball_scores


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
