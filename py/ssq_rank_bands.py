"""Score-rank band definitions shared by selection and reporting."""

from ssq_config import DEFAULT_STRATEGY_CONFIG
from ssq_core import RED_BALLS

RANK_BAND_NAMES = ('high', 'middle', 'low', 'other')


def build_rank_bands(config=DEFAULT_STRATEGY_CONFIG):
    """Return score-rank bands matching the configured mixed pool."""
    total = len(RED_BALLS)
    middle_count = config.pool_size_red - config.high_count - config.low_count
    available_count = total - config.high_count - config.low_count
    middle_offset = max(0, (available_count - middle_count) // 2)
    middle_start = config.high_count + middle_offset + 1
    return {
        'high': range(1, config.high_count + 1),
        'middle': range(middle_start, middle_start + middle_count),
        'low': range(total - config.low_count + 1, total + 1),
    }


def build_rank_band_widths(config=DEFAULT_STRATEGY_CONFIG):
    bands = build_rank_bands(config)
    return {
        **{name: len(ranks) for name, ranks in bands.items()},
        'other': len(RED_BALLS) - sum(len(ranks) for ranks in bands.values()),
    }


def build_rank_band_labels(config=DEFAULT_STRATEGY_CONFIG):
    bands = build_rank_bands(config)

    def describe(title, ranks):
        return title if not ranks else f'{title}({ranks.start}-{ranks.stop - 1})'

    return {
        'high': describe('高端', bands['high']),
        'middle': describe('中段', bands['middle']),
        'low': describe('低端', bands['low']),
        'other': '其他',
    }


RANK_BANDS = build_rank_bands()
RANK_BAND_WIDTHS = build_rank_band_widths()
