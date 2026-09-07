"""Runtime metadata collection and atomic analysis report persistence."""

import os
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

from ssq_file_io import atomic_write_text
from ssq_report_builder import build_analysis_report

RUNTIME_PACKAGES = ('numpy', 'pandas', 'lightgbm', 'scikit-learn')


@dataclass(frozen=True)
class ReportOutputDependencies:
    build_report: Callable[..., str]
    write_text: Callable[..., Any]


def default_output_dependencies():
    return ReportOutputDependencies(
        build_report=build_analysis_report,
        write_text=atomic_write_text,
    )


def collect_runtime_versions(packages: Sequence[str] = RUNTIME_PACKAGES):
    return {
        'python': platform.python_version(),
        **{package: version(package) for package in packages},
    }


def save_analysis_report(data, report_dir, dependencies=None):
    """Build, print, and atomically persist an analysis report."""
    if dependencies is None:
        dependencies = default_output_dependencies()
    report = dependencies.build_report(data)
    print('\n\n' + report)
    try:
        os.makedirs(report_dir, exist_ok=True)
        timestamp = data.generated_at.strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(report_dir, f'ssq_analysis_output_{timestamp}.txt')
        dependencies.write_text(filepath, report)
    except OSError as exc:
        raise SystemExit(f'\n\n写入报告文件失败: {exc}') from exc
    print(f'\n\n报告已成功保存到文件: {filepath}')
    return filepath
