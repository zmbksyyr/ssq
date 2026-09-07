import unittest

import ssq_selection as selection
import ssq_selection_models as models


class SelectionModelTests(unittest.TestCase):
    def test_selection_preserves_model_compatibility_exports(self):
        self.assertIs(
            selection.RedCandidateSelection,
            models.RedCandidateSelection,
        )
        self.assertIs(
            selection.CandidateGenerationRequest,
            models.CandidateGenerationRequest,
        )
        self.assertIs(
            selection.DuplexSelectionRequest,
            models.DuplexSelectionRequest,
        )

    def test_duplex_context_defaults_are_not_shared(self):
        first = models.DuplexSelectionRequest((), tuple(range(1, 8)))
        second = models.DuplexSelectionRequest((), tuple(range(1, 8)))

        self.assertIsNot(first.context, second.context)


if __name__ == '__main__':
    unittest.main()
