"""Shared domain rules for the Double Color Ball lottery."""

import os
import tempfile

import ssq_domain as _domain
import ssq_draw_schedule as _draw_schedule
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


local_now = _draw_schedule.local_now
local_today = _draw_schedule.local_today
parse_issue = _parsing.parse_issue
parse_red_balls = _parsing.parse_red_balls
parse_blue_ball = _parsing.parse_blue_ball
parse_blue_balls = _parsing.parse_blue_balls
validate_ball_scores = _parsing.validate_ball_scores
infer_next_issue = _draw_schedule.infer_next_issue


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
