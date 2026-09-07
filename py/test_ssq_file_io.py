import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ssq_core
import ssq_file_io as file_io


class AtomicFileIoTests(unittest.TestCase):
    def test_core_preserves_atomic_write_compatibility_export(self):
        self.assertIs(ssq_core.atomic_write_text, file_io.atomic_write_text)

    def test_atomic_write_creates_parent_and_replaces_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'report.txt'

            file_io.atomic_write_text(path, '新内容')

            self.assertEqual(path.read_text(encoding='utf-8'), '新内容')
            self.assertEqual(list(path.parent.glob('.ssq-*.tmp')), [])

    def test_writer_failure_preserves_target_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.txt'
            path.write_text('原内容', encoding='utf-8')

            with (
                self.assertRaisesRegex(RuntimeError, 'write failed'),
                file_io.atomic_text_writer(path) as temporary_file,
            ):
                temporary_file.write('不完整内容')
                raise RuntimeError('write failed')

            self.assertEqual(path.read_text(encoding='utf-8'), '原内容')
            self.assertEqual(list(path.parent.glob('.ssq-*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
