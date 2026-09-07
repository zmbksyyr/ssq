"""Shared domain constants for the Double Color Ball lottery."""

from zoneinfo import ZoneInfo

RED_MIN = 1
RED_MAX = 33
BLUE_MIN = 1
BLUE_MAX = 16
RED_COUNT = 6
RED_BALLS = tuple(range(RED_MIN, RED_MAX + 1))
BLUE_BALLS = tuple(range(BLUE_MIN, BLUE_MAX + 1))
PRIME_RED_BALLS = frozenset({2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31})
DRAW_COLUMNS = ('期号', '日期', '红球', '蓝球')
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
