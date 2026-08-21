"""Generate or verify the local evidence record used by the JOSS manuscript.

Run from any directory with ``python scripts/joss_audit.py``. Write mode updates only
``docs/audit/joss_evidence.json``; ``--check`` writes nothing and exits non-zero when the tracked
record differs from the current checkout. Git metrics are intentionally checked only by this
manual command because they move on every commit. Installed-wheel tests enforce the stable live
metrics separately.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_PATH = REPOSITORY_ROOT / "docs" / "audit" / "joss_evidence.json"
PAPER_PATH = REPOSITORY_ROOT / "paper.md"
API_PATH = REPOSITORY_ROOT / "docs" / "api.md"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"


def _git(*arguments: str) -> str:
    """Return stripped output from a Git command executed at the repository root."""
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _paper_body(text: str) -> str:
    """Return manuscript content after the YAML front matter."""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                return "\n".join(lines[index + 1 :])
    return text


def _collected_test_count() -> int:
    """Collect and return the number of pytest cases without running them."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = re.search(r"(\d+) tests? collected", completed.stdout)
    if summary is not None:
        return int(summary.group(1))
    file_counts = re.findall(r"\.py:\s+(\d+)$", completed.stdout, flags=re.MULTILINE)
    if not file_counts:
        raise RuntimeError("pytest collection completed but its test count could not be parsed")
    return sum(int(value) for value in file_counts)


def measure() -> dict[str, Any]:
    """Measure the live software, manuscript, and local Git facts used in JOSS claims."""
    import stochvolmodels

    paper = PAPER_PATH.read_text(encoding="utf-8")
    api = API_PATH.read_text(encoding="utf-8")
    project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))["project"]
    stable_exports = list(stochvolmodels.__all__)
    commit_months = _git("log", "--format=%ad", "--date=format:%Y-%m").splitlines()
    release_tags = _git("tag", "--list", "v*").splitlines()

    return {
        "measured_at_commit": _git("rev-parse", "HEAD"),
        "metrics": {
            "package_version": {
                "value": project["version"],
                "how": "project.version in pyproject.toml",
                "live": True,
                "paper_phrase": None,
            },
            "stable_root_exports": {
                "value": len(stable_exports),
                "how": "len(stochvolmodels.__all__)",
                "live": True,
                "paper_phrase": f"has {len(stable_exports)} explicit exports",
            },
            "stable_exports_rendered_in_api": {
                "value": sum(name in api for name in stable_exports),
                "how": "stable __all__ names present in docs/api.md",
                "live": True,
                "paper_phrase": None,
            },
            "paper_body_words": {
                "value": len(_paper_body(paper).split()),
                "how": "whitespace-separated paper.md body tokens after YAML front matter",
                "live": True,
                "paper_phrase": None,
            },
            "collected_tests": {
                "value": _collected_test_count(),
                "how": "python -m pytest --collect-only -q",
                "live": False,
                "paper_phrase": None,
            },
            "commits": {
                "value": int(_git("rev-list", "--count", "HEAD")),
                "how": "git rev-list --count HEAD",
                "live": False,
                "paper_phrase": None,
            },
            "active_calendar_months": {
                "value": len(set(commit_months)),
                "how": "distinct commit YYYY-MM values through measured_at_commit",
                "live": False,
                "paper_phrase": f"activity in {len(set(commit_months))} distinct calendar months",
            },
            "first_commit_date": {
                "value": _git("log", "--reverse", "--format=%ad", "--date=short").splitlines()[0],
                "how": "first local Git commit date",
                "live": False,
                "paper_phrase": None,
            },
            "release_tags": {
                "value": len(release_tags),
                "how": "Git tags matching v* through measured_at_commit",
                "live": False,
                "paper_phrase": None,
            },
        },
    }


def build_record() -> dict[str, Any]:
    """Preserve immutable baseline evidence and attach a fresh current measurement."""
    existing = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    existing["current"] = measure()
    return existing


def main() -> int:
    """Write the evidence record, or compare it without writing under ``--check``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare without writing")
    arguments = parser.parse_args()
    measured = build_record()
    rendered = json.dumps(measured, indent=2, ensure_ascii=False) + "\n"

    if arguments.check:
        tracked = EVIDENCE_PATH.read_text(encoding="utf-8")
        if tracked != rendered:
            print("JOSS evidence is stale; run python scripts/joss_audit.py", file=sys.stderr)
            return 1
        print("JOSS evidence matches the current checkout")
        return 0

    EVIDENCE_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {EVIDENCE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
