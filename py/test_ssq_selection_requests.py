import unittest

from ssq_core import RED_BALLS
from ssq_rules import (
    RecommendationRequest,
    RuleContext,
    select_recommendation_portfolio,
)
from ssq_selection import (
    CandidateGenerationRequest,
    DuplexSelectionRequest,
    generate_candidates,
    rank_duplex_candidates,
)


def red_scores():
    return {ball: float(ball) for ball in RED_BALLS}


class SelectionRequestValidationTests(unittest.TestCase):
    def test_candidate_request_rejects_invalid_controls(self):
        defaults = {
            'red_scores': red_scores(),
            'context': RuleContext(),
            'rejection_set': set(),
        }
        invalid = (
            {'context': object()},
            {'config': object()},
            {'mode': 'unknown'},
            {'show_progress': 1},
            {'rejection_set': iter(())},
        )

        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(
                (TypeError, ValueError)
            ):
                generate_candidates(CandidateGenerationRequest(
                    **{**defaults, **changes}
                ))

    def test_duplex_request_rejects_inconsistent_collections(self):
        defaults = {
            'passed_combos': ((1, 2, 3, 4, 5, 6),),
            'red_pool': tuple(range(1, 8)),
        }
        invalid = (
            {'context': object()},
            {'red_pool': (1, 2, 3, 4, 5, 6)},
            {'red_pool': (1, 2, 3, 4, 5, 6, 34)},
            {'passed_combos': iter(defaults['passed_combos'])},
            {'passed_combos': ((1, 2, 3, 4, 5, 8),)},
            {'passed_combos': defaults['passed_combos'] * 2},
            {'red_scores': {}},
        )

        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(
                (TypeError, ValueError)
            ):
                rank_duplex_candidates(DuplexSelectionRequest(
                    **{**defaults, **changes}
                ))

    def test_recommendation_request_normalizes_generator_once(self):
        combos = ((1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 7))
        request = RecommendationRequest(
            passed_combos=(combo for combo in combos),
            red_scores=red_scores(),
            context=RuleContext(),
            limit=1,
            max_shared=4,
        )

        result = select_recommendation_portfolio(request)

        self.assertEqual(len(result), 1)
        self.assertIn(result[0], combos)
        self.assertEqual(tuple(request.passed_combos), ())

    def test_recommendation_request_rejects_invalid_controls(self):
        defaults = {
            'passed_combos': ((1, 2, 3, 4, 5, 6),),
            'red_scores': red_scores(),
            'context': RuleContext(),
        }
        invalid = (
            {'context': object()},
            {'limit': True},
            {'limit': -1},
            {'max_shared': 4.5},
            {'passed_combos': defaults['passed_combos'] * 2},
            {'passed_combos': ((1, 2, 3, 4, 5, 34),)},
            {'red_scores': {}},
        )

        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(
                (TypeError, ValueError)
            ):
                select_recommendation_portfolio(RecommendationRequest(
                    **{**defaults, **changes}
                ))


if __name__ == '__main__':
    unittest.main()
