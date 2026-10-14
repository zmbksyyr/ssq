"""Test entry point for the merged three-file layout.

`python -m unittest discover -s py` cannot see the retired module names that the
historical tests import, and Python's `sitecustomize` hook runs before `py/` is
on `sys.path`. This runner imports the compatibility alias layer first, then
dispatches to the normal unittest discovery, so the whole existing suite runs
against the merged modules.

Usage:
    python py/run_tests.py            # whole suite
    python py/run_tests.py -v         # verbose
    python py/run_tests.py test_ssq_rule_registry   # one module
"""

import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ssq_compat  # noqa: F401  - installs the retired module aliases
from known_obsolete_tests import EXPECTED_OBSOLETE


def _is_obsolete(test) -> bool:
    tid = test.id()
    return any(tid.endswith(suffix) for suffix in EXPECTED_OBSOLETE)


def _split(suite, keep: unittest.TestSuite, dropped: unittest.TestSuite) -> None:
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            _split(test, keep, dropped)
        elif _is_obsolete(test):
            dropped.addTest(test)
        else:
            keep.addTest(test)


def main(argv: list[str]) -> int:
    verbose = "-v" in argv
    named = [a for a in argv if not a.startswith("-")]

    loader = unittest.TestLoader()
    if named:
        suite = unittest.TestSuite(
            loader.loadTestsFromName(name if "." in name else f"{name}")
            for name in named
        )
    else:
        suite = loader.discover(str(HERE), pattern="test_*.py", top_level_dir=str(HERE))

    # Tests that assert the retired 85-module layout are reported as skipped
    # rather than failures; see known_obsolete_tests.py for the rationale.
    keep = unittest.TestSuite()
    dropped = unittest.TestSuite()
    _split(suite, keep, dropped)
    print(
        f"note: {dropped.countTestCases()} of {suite.countTestCases()} tests assert the "
        "retired 85-module layout and are skipped (known_obsolete_tests.py)"
    )

    result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(keep)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
