import unittest

import ssq_model_contracts as contracts
import ssq_training as training


class ModelContractTests(unittest.TestCase):
    def test_training_preserves_model_contract_compatibility_exports(self):
        self.assertIs(training.BallModelSpec, contracts.BallModelSpec)
        self.assertIs(
            training.MODEL_TRAINING_PARAMS,
            contracts.MODEL_TRAINING_PARAMS,
        )

    def test_model_spec_is_immutable(self):
        spec = contracts.BallModelSpec(
            candidates=(1,),
            outcome_column='红球',
            contains_candidate=lambda draw, ball: ball in draw,
        )

        with self.assertRaises((AttributeError, TypeError)):
            spec.outcome_column = '蓝球'

    def test_training_parameters_are_deterministic_and_immutable(self):
        params = contracts.MODEL_TRAINING_PARAMS

        self.assertEqual(params['random_state'], 42)
        self.assertTrue(params['deterministic'])
        self.assertTrue(params['force_col_wise'])
        with self.assertRaises(TypeError):
            params['random_state'] = 7


if __name__ == '__main__':
    unittest.main()
