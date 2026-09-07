import io
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import ssq_report_output as report_output
import ssq_workflow as workflow


class ReportOutputTests(unittest.TestCase):
    def test_workflow_preserves_runtime_version_compatibility_export(self):
        self.assertIs(
            workflow.collect_runtime_versions,
            report_output.collect_runtime_versions,
        )

    def test_report_is_built_and_written_with_timestamped_name(self):
        data = SimpleNamespace(
            generated_at=datetime(2026, 9, 7, 12, 34, 56, tzinfo=timezone.utc),
        )
        dependencies = report_output.ReportOutputDependencies(
            build_report=Mock(return_value='analysis report'),
            write_text=report_output.atomic_write_text,
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch('sys.stdout', new_callable=io.StringIO),
        ):
            filepath = report_output.save_analysis_report(
                data,
                directory,
                dependencies,
            )
            content = Path(filepath).read_text(encoding='utf-8')

        self.assertEqual(Path(filepath).name, 'ssq_analysis_output_20260907_123456.txt')
        self.assertEqual(content, 'analysis report')
        dependencies.build_report.assert_called_once_with(data)

    def test_write_failure_becomes_workflow_exit(self):
        data = SimpleNamespace(
            generated_at=datetime(2026, 9, 7, 12, 34, 56, tzinfo=timezone.utc),
        )
        dependencies = report_output.ReportOutputDependencies(
            build_report=Mock(return_value='analysis report'),
            write_text=Mock(side_effect=OSError('disk full')),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch('sys.stdout', new_callable=io.StringIO),
            self.assertRaisesRegex(SystemExit, '写入报告文件失败: disk full'),
        ):
            report_output.save_analysis_report(data, directory, dependencies)


if __name__ == '__main__':
    unittest.main()
