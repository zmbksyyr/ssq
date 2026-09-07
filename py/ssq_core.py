"""Shared domain rules for the Double Color Ball lottery."""

RED_MIN = 1
RED_MAX = 33
BLUE_MIN = 1
BLUE_MAX = 16
RED_COUNT = 6

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


def parse_red_balls(value, expected_count=RED_COUNT):
    if isinstance(value, str):
        numbers = [int(part.strip()) for part in value.split(',')]
    else:
        numbers = [int(part) for part in value]
    if len(numbers) != expected_count:
        raise ValueError(f"红球数量必须为 {expected_count}，实际为 {len(numbers)}")
    if len(set(numbers)) != expected_count:
        raise ValueError("红球号码不能重复")
    if any(number < RED_MIN or number > RED_MAX for number in numbers):
        raise ValueError(f"红球号码必须在 {RED_MIN} 到 {RED_MAX} 之间")
    return sorted(numbers)


def parse_blue_ball(value):
    number = int(value)
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
