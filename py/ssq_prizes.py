"""Bet parsing and fixed-reference prize calculations."""

import re
from math import comb

from ssq_core import parse_blue_ball, parse_blue_balls, parse_red_balls
from ssq_domain import PRIZE_NAMES, PRIZE_RULES

SINGLE_HEADER_PATTERN = re.compile(r'^【单式推荐 \((\d+)组\)】$', re.MULTILINE)
DUPLEX_HEADER_PATTERN = re.compile(r'^【7\+N 复式推荐 \(1组\)】$', re.MULTILINE)
SINGLE_BET_PATTERN = re.compile(
    r'^组合\s+\d+:\s+红球\s+\[(.*?)\]\s+蓝球\s+\[(.*?)\]$'
)


def parse_single_bet_line(line):
    match = SINGLE_BET_PATTERN.fullmatch(line)
    if match is None:
        raise ValueError('字段格式不完整')
    return {
        'red': parse_red_balls(match.group(1)),
        'blue': parse_blue_ball(match.group(2)),
    }


def parse_duplex_section(section):
    red_match = re.search(r'^\s*红球:\s*\[(.*?)\]\s*$', section, re.MULTILINE)
    blue_match = re.search(r'^\s*蓝球:\s*\[(.*?)\]\s*$', section, re.MULTILINE)
    if red_match is None or blue_match is None:
        raise ValueError('复式投注内容不完整')
    return {
        'red': parse_red_balls(red_match.group(1), expected_count=7),
        'blue': parse_blue_balls(blue_match.group(1)),
    }


def validate_parsed_bets(expected_count, single_bets):
    if len(single_bets) != expected_count:
        raise ValueError(
            f'单式投注数量不完整: 声明 {expected_count} 注，'
            f'实际解析 {len(single_bets)} 注'
        )
    unique_singles = {
        (tuple(bet['red']), bet['blue']) for bet in single_bets
    }
    if len(unique_singles) != len(single_bets):
        raise ValueError('报告包含重复单式投注')


def parse_report_bets(filepath):
    """Parse a complete set of single and duplex bets from an analysis report."""
    with open(filepath, encoding='utf-8') as report_file:
        content = report_file.read()
    single_header = SINGLE_HEADER_PATTERN.search(content)
    if single_header is None:
        raise ValueError('报告缺少或无法识别单式推荐标题')
    duplex_header = DUPLEX_HEADER_PATTERN.search(content, single_header.end())
    if duplex_header is None:
        raise ValueError('报告缺少或无法识别复式推荐标题')

    single_section = content[single_header.end():duplex_header.start()]
    single_lines = [
        line.strip() for line in single_section.splitlines()
        if line.strip().startswith('组合')
    ]
    try:
        single_bets = [parse_single_bet_line(line) for line in single_lines]
    except (TypeError, ValueError) as exc:
        raise ValueError(f'单式投注解析失败: {exc}') from exc
    validate_parsed_bets(int(single_header.group(1)), single_bets)

    try:
        duplex_bet = parse_duplex_section(content[duplex_header.end():])
    except (TypeError, ValueError) as exc:
        raise ValueError(f'复式投注解析失败: {exc}') from exc
    return single_bets, duplex_bet


def calculate_single_prize(bet_reds, bet_blue, winning_reds, winning_blue):
    """Calculate the fixed-reference prize for one standard ticket."""
    bet_red_set = set(parse_red_balls(bet_reds))
    winning_red_set = set(parse_red_balls(winning_reds))
    bet_blue = parse_blue_ball(bet_blue)
    winning_blue = parse_blue_ball(winning_blue)
    red_hits = len(bet_red_set & winning_red_set)
    blue_hit = int(bet_blue == winning_blue)
    hit_key = (red_hits, blue_hit)
    return (
        PRIZE_RULES.get(hit_key, 0),
        PRIZE_NAMES.get(hit_key, '未中奖'),
        f'命中{red_hits}+{blue_hit}',
    )


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
        red_combos = (
            comb(red_hits, red_needed) * comb(red_misses, 6 - red_needed)
        )
        blue_combos = (
            blue_hit if blue_needed else len(unique_blues) - blue_hit
        )
        winning_tickets = red_combos * blue_combos
        if winning_tickets:
            prize_name = PRIZE_NAMES[(red_needed, blue_needed)]
            prize_breakdown[prize_name] = (
                prize_breakdown.get(prize_name, 0) + winning_tickets
            )
            total_prize += winning_tickets * prize_value

    summary = f'总计命中 {red_hits} 个红球, {blue_hit} 个蓝球'
    return total_prize, prize_breakdown, summary
