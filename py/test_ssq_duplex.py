import unittest

import ssq_duplex as duplex
import ssq_selection as selection


class DuplexModuleTests(unittest.TestCase):
    def test_selection_preserves_duplex_compatibility_exports(self):
        self.assertIs(
            selection.find_best_7_red_combinations,
            duplex.find_best_7_red_combinations,
        )
        self.assertIs(
            selection.rank_duplex_candidates,
            duplex.rank_duplex_candidates,
        )
        self.assertIs(
            selection.validate_duplex_selection_request,
            duplex.validate_duplex_selection_request,
        )

    def test_equal_duplex_scores_use_stable_number_order(self):
        request = selection.DuplexSelectionRequest(
            passed_combos=((1, 2, 3, 4, 5, 6),),
            red_pool=tuple(range(1, 9)),
        )

        ranked = duplex.rank_duplex_candidates(request)

        self.assertEqual(ranked[0], ((1, 2, 3, 4, 5, 6, 7), 1))
        self.assertEqual(ranked[1], ((1, 2, 3, 4, 5, 6, 8), 1))


if __name__ == '__main__':
    unittest.main()
