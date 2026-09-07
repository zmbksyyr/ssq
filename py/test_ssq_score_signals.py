import unittest

import pandas as pd
import ssq_score_signals as signals
import ssq_scoring as scoring


class ScoreSignalTests(unittest.TestCase):
    def test_scoring_preserves_signal_compatibility_exports(self):
        self.assertIs(scoring.get_omission, signals.get_omission)
        self.assertIs(
            scoring.get_weighted_frequency,
            signals.get_weighted_frequency,
        )

    def test_omission_tracks_last_seen_draw_and_never_seen_balls(self):
        history = pd.DataFrame({'红球': [
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8],
        ]})

        omission = signals.get_omission(history)

        self.assertEqual(omission[1], 2)
        self.assertEqual(omission[2], 1)
        self.assertEqual(omission[3], 0)
        self.assertEqual(omission[33], 3)

    def test_weighted_frequency_favors_more_recent_occurrences(self):
        draws = pd.Series([[1], [2]])

        frequency = signals.get_weighted_frequency(draws, decay_factor=0.5)

        self.assertEqual(frequency[1], 0.5)
        self.assertEqual(frequency[2], 1.0)


if __name__ == '__main__':
    unittest.main()
