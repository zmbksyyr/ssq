import json
import tempfile
import unittest
from pathlib import Path

import ssq_strategy_params as strategy_params
import ssq_workflow as workflow
from ssq_config import DEFAULT_PARAMS


class StrategyParamTests(unittest.TestCase):
    def test_missing_file_returns_validated_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missing.json'

            loaded = strategy_params.load_strategy_params(path)

        self.assertFalse(loaded.loaded_from_file)
        self.assertEqual(loaded.values, DEFAULT_PARAMS)

    def test_valid_and_invalid_json_are_distinguished(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'params.json'
            path.write_text('{invalid', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '参数文件'):
                strategy_params.load_strategy_params(path)

            path.write_text(json.dumps(DEFAULT_PARAMS), encoding='utf-8')
            loaded = strategy_params.load_strategy_params(path)

        self.assertTrue(loaded.loaded_from_file)
        self.assertEqual(loaded.values, DEFAULT_PARAMS)

    def test_workflow_wrapper_delegates_explicit_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missing.json'

            loaded = workflow.load_strategy_params(path)

        self.assertFalse(loaded.loaded_from_file)


if __name__ == '__main__':
    unittest.main()
