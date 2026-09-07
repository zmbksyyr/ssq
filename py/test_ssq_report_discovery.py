import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ssq_bonus_workflow as workflow
import ssq_report_discovery as discovery


class ReportDiscoveryTests(unittest.TestCase):
    def test_workflow_preserves_metadata_parser_export(self):
        self.assertIs(
            workflow.parse_report_target_issue,
            discovery.parse_report_target_issue,
        )

    def test_target_issue_metadata_must_be_unique_and_valid(self):
        self.assertEqual(
            discovery.parse_report_target_issue(
                'Data_Basis_Issue: 2026001\n'
                'Prediction_Target_Issue: 2026002\n'
            ),
            2026002,
        )
        for content in (
            'Data_Basis_Issue: 2026001\n',
            'Prediction_Target_Issue: 999\n',
            (
                'Prediction_Target_Issue: 2026001\n'
                'Prediction_Target_Issue: 2026002\n'
            ),
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                discovery.parse_report_target_issue(content)

    def test_discovery_skips_newer_invalid_and_wrong_issue_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory)
            matching = report_dir / 'ssq_analysis_output_20260101_000000.txt'
            wrong = report_dir / 'ssq_analysis_output_20260102_000000.txt'
            invalid = report_dir / 'ssq_analysis_output_20260103_000000.txt'
            matching.write_text(
                'Prediction_Target_Issue: 2026001\n', encoding='utf-8'
            )
            wrong.write_text(
                'Prediction_Target_Issue: 2026002\n', encoding='utf-8'
            )
            invalid.write_text(
                'Prediction_Target_Issue: 2026001\n'
                'Prediction_Target_Issue: 2026002\n',
                encoding='utf-8',
            )

            result, error = discovery.find_matching_report(2026001, report_dir)

        self.assertIsNone(error)
        self.assertEqual(Path(result), matching)

    def test_discovery_distinguishes_empty_directory_from_no_match(self):
        with tempfile.TemporaryDirectory() as directory:
            result, error = discovery.find_matching_report(2026001, directory)
            self.assertIsNone(result)
            self.assertIn('未找到任何', error)

            path = Path(directory) / 'ssq_analysis_output_20260101_000000.txt'
            path.write_text(
                'Prediction_Target_Issue: 2026002\n', encoding='utf-8'
            )
            result, error = discovery.find_matching_report(2026001, directory)

        self.assertIsNone(result)
        self.assertIn('未找到预测目标期号', error)


if __name__ == '__main__':
    unittest.main()
