import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ssq_data_store
import ssq_history_store as history_store


class HistoryStoreTests(unittest.TestCase):
    def test_data_store_preserves_history_compatibility_export(self):
        self.assertIs(ssq_data_store.update_csv_file, history_store.update_csv_file)

    def test_merge_workflow_uses_injected_dependencies(self):
        normalize = Mock(side_effect=lambda frame: frame)
        read = Mock(return_value=pd.DataFrame())
        validate = Mock()
        write = Mock()
        dependencies = history_store.HistoryStoreDependencies(
            normalize_frame=normalize,
            read_csv=read,
            validate_snapshot=validate,
            write_csv=write,
        )
        records = [{
            '期号': '2026001',
            '日期': '2026-01-01',
            '红球': '01,02,03,04,05,06',
            '蓝球': '07',
        }]

        updated = history_store.update_csv_file(
            'draws.csv',
            records,
            require_full_snapshot=True,
            dependencies=dependencies,
        )

        self.assertTrue(updated)
        read.assert_called_once_with('draws.csv')
        validate.assert_called_once()
        self.assertEqual(normalize.call_count, 2)
        written_frame, written_path = write.call_args.args
        self.assertEqual(written_frame.to_dict('records'), records)
        self.assertEqual(written_path, 'draws.csv')

    def test_production_modules_do_not_depend_on_compatibility_facade(self):
        for filename in ('ssq_data_workflow.py', 'ssq_data_processor.py'):
            path = Path(__file__).with_name(filename)
            tree = ast.parse(path.read_text(encoding='utf-8'))
            imports = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            with self.subTest(filename=filename):
                self.assertNotIn('ssq_data_store', imports)


if __name__ == '__main__':
    unittest.main()
