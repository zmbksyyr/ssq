import unittest

import ssq_features as features
import ssq_modeling as modeling


class FeatureModuleTests(unittest.TestCase):
    def test_modeling_preserves_feature_compatibility_exports(self):
        self.assertIs(modeling.FEATURE_COLUMNS, features.FEATURE_COLUMNS)
        self.assertIs(modeling.feature_engineer, features.feature_engineer)
        self.assertIs(
            modeling.validate_feature_columns,
            features.validate_feature_columns,
        )

    def test_feature_contract_contains_only_lagged_rolling_inputs(self):
        rolling_columns = {
            column for column in features.FEATURE_COLUMNS
            if column.endswith('_ma5')
        }

        self.assertEqual(
            rolling_columns,
            {'red_sum_ma5', 'odd_count_ma5', 'blue_ma5'},
        )


if __name__ == '__main__':
    unittest.main()
