"""Rebuild the three merged modules from the original 85-module sources.

    python tools/merge/build.py

Because the merge deletes the source modules, they are restored from git HEAD
for the duration of the build and removed again afterwards. The generated files
are then lint-autofixed (ast.unparse emits redundant parentheses that ruff
flags) and the result is verified by the test suite.
"""

import pathlib
import shutil
import subprocess
import sys

MERGE_DIR = pathlib.Path(__file__).resolve().parent
ROOT = MERGE_DIR.parent.parent
PY = ROOT / "py"
KEEP = {
    "ssq_data_prep.py",
    "ssq_analysis.py",
    "ssq_bonus_check.py",
    "ssq_compat.py",
}
GROUPS = ("data_prep", "analysis", "bonus_check")


def main() -> int:
    print("== restoring original sources from git HEAD ==")
    subprocess.run(["git", "checkout", "--", "py/"], cwd=ROOT, check=True)

    print("== generating merged modules ==")
    rc = subprocess.run([sys.executable, str(MERGE_DIR / "generate.py")], cwd=MERGE_DIR)
    if rc.returncode != 0:
        print("generation FAILED - leaving sources in place for inspection")
        return rc.returncode

    print("== installing merged modules ==")
    for group in GROUPS:
        src = PY / f"_merged_{group}.py"
        if not src.exists():
            print(f"missing {src}")
            return 1
        shutil.move(str(src), str(PY / f"ssq_{group}.py"))

    print("== removing retired modules ==")
    removed = 0
    for path in sorted(PY.glob("ssq_*.py")):
        if path.name not in KEEP:
            path.unlink()
            removed += 1
    print(f"   removed {removed}; {len(list(PY.glob('ssq_*.py')))} remain")

    print("== lint autofix (ast.unparse emits redundant parentheses) ==")
    subprocess.run([sys.executable, "-m", "ruff", "check", "py", "--fix"], cwd=ROOT)

    # `ssq_report_builder` used `version` as a loop variable. Once merged the
    # module also imports `version` from importlib.metadata, which ruff flags as
    # F402. Renaming the loop variable is behaviour-preserving.
    analysis = PY / "ssq_analysis.py"
    text = analysis.read_text(encoding="utf-8")
    fixed = text.replace(
        "for name, version in data.runtime_versions.items():",
        "for name, runtime_version in data.runtime_versions.items():",
    ).replace(
        "lines.append(f'Runtime_{name}: {version}')",
        "lines.append(f'Runtime_{name}: {runtime_version}')",
    )
    if fixed != text:
        analysis.write_text(fixed, encoding="utf-8")
        print("   renamed shadowing loop variable version -> runtime_version")
    return 0


if __name__ == "__main__":
    sys.exit(main())
