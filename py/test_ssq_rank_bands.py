import unittest

import ssq_rank_bands as rank_bands
import ssq_selection as selection
from ssq_config import StrategyConfig


class RankBandTests(unittest.TestCase):
    def test_default_rank_bands_preserve_candidate_mix(self):
        self.assertEqual(
            rank_bands.RANK_BAND_WIDTHS,
            {'high': 4, 'middle': 9, 'low': 4, 'other': 16},
        )
        self.assertEqual(rank_bands.RANK_BANDS['high'], range(1, 5))
        self.assertEqual(rank_bands.RANK_BANDS['middle'], range(13, 22))
        self.assertEqual(rank_bands.RANK_BANDS['low'], range(30, 34))

    def test_custom_rank_bands_preserve_ranges_and_labels(self):
        config = StrategyConfig(pool_size_red=10, high_count=2, low_count=3)

        self.assertEqual(
            rank_bands.build_rank_bands(config),
            {
                'high': range(1, 3),
                'middle': range(14, 19),
                'low': range(31, 34),
            },
        )
        self.assertEqual(
            rank_bands.build_rank_band_widths(config),
            {'high': 2, 'middle': 5, 'low': 3, 'other': 23},
        )
        self.assertEqual(
            rank_bands.build_rank_band_labels(config),
            {
                'high': '高端(1-2)',
                'middle': '中段(14-18)',
                'low': '低端(31-33)',
                'other': '其他',
            },
        )

    def test_selection_compatibility_exports_match_shared_definitions(self):
        config = StrategyConfig(pool_size_red=10, high_count=2, low_count=3)

        self.assertEqual(selection.RANK_BANDS, rank_bands.RANK_BANDS)
        self.assertEqual(selection.RANK_BAND_WIDTHS, rank_bands.RANK_BAND_WIDTHS)
        self.assertEqual(
            selection.build_rank_bands(config),
            rank_bands.build_rank_bands(config),
        )
        self.assertEqual(
            selection.build_rank_band_widths(config),
            rank_bands.build_rank_band_widths(config),
        )
        self.assertEqual(
            selection.build_rank_band_labels(config),
            rank_bands.build_rank_band_labels(config),
        )


if __name__ == '__main__':
    unittest.main()
