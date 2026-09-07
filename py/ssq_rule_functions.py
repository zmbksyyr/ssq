"""Pure red-ball rule predicates and balance scores."""

from collections import Counter
from itertools import combinations

from ssq_core import PRIME_RED_BALLS


def is_prime(number):
    return number in PRIME_RED_BALLS


def calculate_ac_value(combo):
    standard_ac = len({abs(left - right) for left, right in combinations(combo, 2)})
    return standard_ac - 5


def filter_highly_regular(combo):
    differences = {
        combo[index + 1] - combo[index] for index in range(len(combo) - 1)
    }
    return len(differences) > 1


def filter_sum_value(combo):
    return 70 <= sum(combo) <= 160


def filter_span(combo):
    return combo[-1] - combo[0] >= 15


def filter_consecutive_numbers(combo):
    groups = 0
    max_length = 0
    current_length = 1
    for index in range(len(combo) - 1):
        if combo[index + 1] - combo[index] == 1:
            current_length += 1
        else:
            if current_length >= 2:
                groups += 1
                max_length = max(max_length, current_length)
            current_length = 1
    if current_length >= 2:
        groups += 1
        max_length = max(max_length, current_length)
    return not (groups >= 3 or max_length >= 4)


def filter_zones(combo):
    all_in_small = all(ball <= 11 for ball in combo)
    all_in_medium = all(12 <= ball <= 22 for ball in combo)
    all_in_large = all(ball >= 23 for ball in combo)
    return not (all_in_small or all_in_medium or all_in_large)


def filter_ac_value(combo):
    return 6 <= calculate_ac_value(combo) <= 10


def filter_prime_composite_ratio(combo):
    prime_count = sum(is_prime(ball) for ball in combo)
    return prime_count not in (0, 1, 5, 6)


def filter_big_small_ratio(combo):
    small_count = sum(ball <= 16 for ball in combo)
    return small_count not in (0, 1, 5, 6)


def filter_recent_overlap(combo, recent_draws):
    candidate = set(combo)
    return all(len(candidate & draw) < 4 for draw in recent_draws)


def filter_all_cold(combo, omission_values):
    return not all(omission_values.get(ball, 0) > 15 for ball in combo)


def filter_odd_even_ratio(combo):
    even_count = sum(ball % 2 == 0 for ball in combo)
    return even_count not in (0, 1, 5, 6)


def filter_modulo3_roads(combo):
    return len({ball % 3 for ball in combo}) == 3


def filter_ending_digits(combo):
    counts = Counter(ball % 10 for ball in combo)
    return max(counts.values()) < 3 and len(counts) > 2


def filter_head_tail_range(combo):
    return combo[0] <= 10 and combo[-1] >= 25


def filter_sum_of_tails(combo):
    return 15 <= sum(ball % 10 for ball in combo) <= 45


def filter_related_numbers(combo, last_draw):
    candidate = set(combo)
    repeats = candidate & last_draw
    adjacent_numbers = (
        {number - 1 for number in last_draw}
        | {number + 1 for number in last_draw}
    )
    return bool(repeats or candidate & adjacent_numbers)


def filter_diagonal_consecutive(combo, last_draw, previous_draw):
    return not any(
        ball - 1 in last_draw and ball - 2 in previous_draw
        for ball in combo
    )


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
