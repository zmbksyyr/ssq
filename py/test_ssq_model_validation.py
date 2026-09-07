import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import ssq_model_validation as validation
import ssq_training as training
from ssq_domain import BLUE_BALLS, RED_BALLS


class ModelValidationTests(unittest.TestCase):
    def test_training_preserves_validation_compatibility_exports(self):
        self.assertIs(training.validate_model_sets, validation.validate_model_sets)
        self.assertIs(
            training.predict_positive_probability,
            validation.predict_positive_probability,
        )

    def test_complete_model_domains_are_accepted(self):
        validation.validate_model_sets(
            {ball: object() for ball in RED_BALLS},
            {ball: object() for ball in BLUE_BALLS},
        )

    def test_probability_requires_one_unique_positive_class(self):
        model = SimpleNamespace(
            classes_=np.array([0, 2]),
            predict_proba=lambda _features: [[0.5, 0.5]],
        )

        with self.assertRaises(ValueError):
            validation.predict_positive_probability(
                model,
                pd.DataFrame({'feature': [1.0]}),
            )


if __name__ == '__main__':
    unittest.main()
