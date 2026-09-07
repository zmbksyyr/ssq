"""Historical frequency and omission signals for lottery balls."""

import numpy as np
import pandas as pd
from ssq_domain import RED_BALLS


def get_omission(df):
    """Calculate how many draws each red ball has been absent."""
    total_draws = len(df)
    last_positions = {}
    for position, draw in enumerate(df['红球']):
        for ball in draw:
            last_positions[ball] = position
    return {
        ball: total_draws - last_positions[ball] - 1
        if ball in last_positions else total_draws
        for ball in RED_BALLS
    }


def get_weighted_frequency(series, decay_factor):
    """Calculate frequency with exponentially greater weight on recent draws."""
    draw_count = len(series)
    weights = np.array([
        decay_factor ** (draw_count - index - 1)
        for index in range(draw_count)
    ])
    weighted_counts = {}
    for index, numbers in enumerate(series):
        for ball in numbers:
            weighted_counts[ball] = weighted_counts.get(ball, 0) + weights[index]
    return pd.Series(weighted_counts)
