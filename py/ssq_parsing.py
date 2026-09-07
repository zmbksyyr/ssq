"""Parsing and validation of lottery domain values."""

from math import isfinite
from numbers import Integral, Real

from ssq_domain import BLUE_MAX, BLUE_MIN, RED_COUNT, RED_MAX, RED_MIN


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
