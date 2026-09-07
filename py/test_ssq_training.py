import unittest

import ssq_modeling as modeling
import ssq_training as training


class TrainingModuleTests(unittest.TestCase):
    def test_modeling_preserves_training_compatibility_exports(self):
        self.assertIs(modeling.BallModelSpec, training.BallModelSpec)
        self.assertIs(
            modeling.MODEL_TRAINING_PARAMS,
            training.MODEL_TRAINING_PARAMS,
        )
        self.assertIs(modeling.train_models_for_spec, training.train_models_for_spec)
        self.assertIs(
            modeling.train_prediction_models,
            training.train_prediction_models,
        )
        self.assertIs(
            modeling.predict_positive_probability,
            training.predict_positive_probability,
        )
        self.assertIs(modeling.lgb, training.lgb)

    def test_training_parameters_remain_deterministic_and_immutable(self):
        self.assertEqual(training.MODEL_TRAINING_PARAMS['random_state'], 42)
        self.assertTrue(training.MODEL_TRAINING_PARAMS['deterministic'])
        with self.assertRaises(TypeError):
            training.MODEL_TRAINING_PARAMS['random_state'] = 7


if __name__ == '__main__':
    unittest.main()
