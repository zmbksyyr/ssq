import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ssq_bonus_workflow as bonus_workflow
import ssq_data_workflow as data_workflow
import ssq_paths as paths
import ssq_workflow as analysis_workflow


class ProjectPathTests(unittest.TestCase):
    def test_project_root_is_derived_from_the_source_directory(self):
        expected_root = Path(__file__).resolve().parent.parent

        self.assertEqual(Path(paths.SCRIPT_DIR), expected_root / 'py')
        self.assertEqual(Path(paths.PROJECT_ROOT), expected_root)
        self.assertEqual(Path(paths.CSV_PATH), expected_root / 'shuangseqiu.csv')
        self.assertEqual(Path(paths.PARAMS_JSON_PATH), expected_root / 'best_params.json')
        self.assertEqual(Path(paths.REPORT_DIR), expected_root / 'report')

    def test_workflows_share_canonical_path_objects(self):
        self.assertIs(analysis_workflow.PROJECT_ROOT, paths.PROJECT_ROOT)
        self.assertIs(analysis_workflow.CSV_PATH, paths.CSV_PATH)
        self.assertIs(analysis_workflow.PARAMS_JSON_PATH, paths.PARAMS_JSON_PATH)
        self.assertIs(analysis_workflow.REPORT_DIR, paths.REPORT_DIR)
        self.assertIs(bonus_workflow.PROJECT_ROOT, paths.PROJECT_ROOT)
        self.assertIs(bonus_workflow.CSV_PATH, paths.CSV_PATH)
        self.assertIs(bonus_workflow.REPORT_DIR, paths.REPORT_DIR)
        self.assertIs(data_workflow.PROJECT_ROOT, paths.PROJECT_ROOT)
        self.assertIs(data_workflow.CSV_FILE_PATH, paths.CSV_PATH)


if __name__ == '__main__':
    unittest.main()
