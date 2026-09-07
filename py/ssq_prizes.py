"""Bet parsing and fixed-reference prize calculations."""

from math import comb

import ssq_bet_parsing as _bet_parsing
from ssq_domain import PRIZE_NAMES, PRIZE_RULES
from ssq_parsing import parse_blue_ball, parse_blue_balls, parse_red_balls

SINGLE_HEADER_PATTERN = _bet_parsing.SINGLE_HEADER_PATTERN
DUPLEX_HEADER_PATTERN = _bet_parsing.DUPLEX_HEADER_PATTERN
SINGLE_BET_PATTERN = _bet_parsing.SINGLE_BET_PATTERN
parse_single_bet_line = _bet_parsing.parse_single_bet_line
parse_duplex_section = _bet_parsing.parse_duplex_section
validate_parsed_bets = _bet_parsing.validate_parsed_bets
parse_report_bets_content = _bet_parsing.parse_report_bets_content
parse_report_bets = _bet_parsing.parse_report_bets


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
