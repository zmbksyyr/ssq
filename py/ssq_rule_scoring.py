"""Soft balance scores used when ranking red-ball combinations."""

from ssq_rule_functions import is_prime


def score_zone_balance(combo):
    counts = (
        sum(ball <= 11 for ball in combo),
        sum(12 <= ball <= 22 for ball in combo),
        sum(ball >= 23 for ball in combo),
    )
    return 1.0 - (max(counts) - min(counts)) / 6


def score_odd_even_balance(combo):
    return 1.0 - abs(sum(ball % 2 for ball in combo) - 3) / 3


def score_prime_balance(combo):
    return 1.0 - abs(sum(is_prime(ball) for ball in combo) - 3) / 3


def score_big_small_balance(combo):
    return 1.0 - abs(sum(ball <= 16 for ball in combo) - 3) / 3
