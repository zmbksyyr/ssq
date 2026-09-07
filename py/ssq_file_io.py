"""Atomic text-file persistence shared by reports and draw data."""

import os
import tempfile
from contextlib import contextmanager


@contextmanager
def atomic_text_writer(path, encoding='utf-8'):
    """Yield an adjacent temporary text file and replace the target on success."""
    target = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding=encoding,
            newline='',
            dir=directory,
            prefix='.ssq-',
            suffix='.tmp',
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            yield temporary_file
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def atomic_write_text(path, content, encoding='utf-8'):
    """Atomically replace a text file with the supplied content."""
    with atomic_text_writer(path, encoding=encoding) as temporary_file:
        temporary_file.write(content)
