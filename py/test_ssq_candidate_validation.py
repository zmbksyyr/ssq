import unittest

import ssq_candidate_validation as validation
import ssq_candidates as candidates
from ssq_config import StrategyConfig
from ssq_rule_models import RuleContext
from ssq_selection_models import CandidateGenerationRequest


class CandidateValidationTests(unittest.TestCase):
    def test_candidates_preserves_validation_compatibility_export(self):
        self.assertIs(
            candidates.validate_candidate_generation_request,
            validation.validate_candidate_generation_request,
        )

    def test_accepts_complete_candidate_request(self):
        request = CandidateGenerationRequest(
            red_scores={ball: float(ball) for ball in range(1, 34)},
            context=RuleContext(),
            rejection_set=set(),
            config=StrategyConfig(),
            mode='mixed',
            show_progress=False,
        )

        self.assertIsNone(validation.validate_candidate_generation_request(request))

    def test_rejects_non_request_object(self):
        with self.assertRaises(TypeError):
            validation.validate_candidate_generation_request(object())


if __name__ == '__main__':
    unittest.main()
