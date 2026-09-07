import random
import unittest

import ssq_anti_crowding as anti_crowding
import ssq_selection as selection
from ssq_config import DEFAULT_STRATEGY_CONFIG


class AntiCrowdingTests(unittest.TestCase):
    def test_selection_preserves_anti_crowding_compatibility_exports(self):
        self.assertIs(
            selection.make_rejection_set,
            anti_crowding.make_rejection_set,
        )
        self.assertIs(
            selection.rejection_seed_for_issue,
            anti_crowding.rejection_seed_for_issue,
        )

    def test_default_size_and_seeded_sample_remain_stable(self):
        self.assertEqual(DEFAULT_STRATEGY_CONFIG.rejection_lib_size, 500_000)
        self.assertEqual(
            anti_crowding.make_rejection_set(8, random.Random(7)),
            {
                (2, 3, 10, 13, 21, 33),
                (2, 4, 6, 19, 27, 28),
                (2, 4, 13, 15, 19, 31),
                (2, 7, 17, 19, 24, 33),
                (3, 5, 10, 15, 18, 28),
                (3, 6, 8, 14, 31, 33),
                (4, 10, 18, 19, 27, 32),
                (7, 12, 19, 21, 31, 32),
            },
        )

    def test_size_and_base_seed_reject_non_integer_values(self):
        for value in (True, 1.5):
            with self.subTest(size=value), self.assertRaises(TypeError):
                anti_crowding.make_rejection_set(value)
            with self.subTest(seed=value), self.assertRaises(TypeError):
                anti_crowding.rejection_seed_for_issue(value, 2026104)


if __name__ == '__main__':
    unittest.main()
