"""Fail-closed repository, packaging, and release-metadata contracts."""

from __future__ import annotations

import ast
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import stochvolmodels

PACKAGE_ROOT = Path(stochvolmodels.__file__).resolve().parent


def _repository_root() -> Path | None:
    """Return the source checkout root, or ``None`` for an installed wheel."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


REPOSITORY_ROOT = _repository_root()


def _has_collected_test(path: Path) -> bool:
    """Return whether a test-shaped module declares a pytest test candidate."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            return True
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            if any(
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name.startswith("test_")
                for child in node.body
            ):
                return True
    return False


def _project_version(pyproject_text: str) -> str:
    """Extract the project version without adding a TOML parser on Python 3.10."""
    project = pyproject_text.split("[project]", maxsplit=1)[-1]
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', project, re.MULTILINE)
    if match is None:
        raise AssertionError("pyproject.toml has no [project] version")
    return match.group(1)


def test_every_shipped_test_shaped_module_is_automated_and_collectable() -> None:
    """Only the central automated suite may use pytest's filename patterns."""
    candidates = sorted(
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if path.name.startswith("test_") or path.name.endswith("_test.py")
        if "pde_solvers" not in path.parts
    )
    assert len(candidates) >= 19, "the shipped automated suite unexpectedly disappeared"

    misplaced = [
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in candidates
        if path.parent != PACKAGE_ROOT / "tests"
    ]
    empty = [
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in candidates
        if not _has_collected_test(path)
    ]
    assert misplaced == [], f"test-shaped manual or misplaced modules: {misplaced}"
    assert empty == [], f"test-shaped modules with no pytest test: {empty}"


def test_core_root_import_does_not_load_optional_or_research_modules() -> None:
    """The public root remains lazy with respect to every optional integration."""
    script = """
import json
import sys
import stochvolmodels

optional = [
    "qis",
    "option_chain_analytics",
    "plotly",
    "sklearn",
    "statsmodels",
    "yfinance",
    "yaml",
]
print(json.dumps({name: name in sys.modules for name in optional}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "qis": False,
        "option_chain_analytics": False,
        "plotly": False,
        "sklearn": False,
        "statsmodels": False,
        "yfinance": False,
        "yaml": False,
    }


def test_imported_version_matches_distribution_metadata() -> None:
    """The lazy package version agrees with the installed distribution."""
    assert stochvolmodels.__version__ == importlib.metadata.version("stochvolmodels")


@pytest.mark.repository_only
@pytest.mark.skipif(REPOSITORY_ROOT is None, reason="repository metadata is absent from the wheel")
def test_release_metadata_versions_and_date_agree() -> None:
    """Pyproject, CFF, README, and the imported distribution stay release-consistent."""
    assert REPOSITORY_ROOT is not None
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    version = _project_version(pyproject)

    cff_version = re.search(r"^version:\s*[\"']?([^\s\"']+)", citation, re.MULTILINE)
    release_date = re.search(r'^date-released:\s*["\']([^"\']+)', citation, re.MULTILINE)
    software_entry = re.search(r"@misc\{sepp\d+stochvolmodels,.*?\n\}", readme, re.DOTALL)

    assert cff_version is not None
    assert cff_version.group(1) == version
    assert release_date is not None
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", release_date.group(1))
    assert software_entry is not None
    assert re.search(rf"version\s*=\s*\{{{re.escape(version)}\}}", software_entry.group(0))
    assert stochvolmodels.__version__ == version


@pytest.mark.repository_only
@pytest.mark.skipif(REPOSITORY_ROOT is None, reason="README and example are absent from the wheel")
def test_readme_first_result_matches_asserted_quickstart_values() -> None:
    """The README's first result remains tied to assertions in the executable quickstart."""
    assert REPOSITORY_ROOT is not None
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = (
        REPOSITORY_ROOT / "examples" / "getting_started" / "quickstart.py"
    ).read_text(encoding="utf-8")
    for value in ("0.197331", "0.275202"):
        assert value in readme
        assert value in quickstart


@pytest.mark.repository_only
@pytest.mark.skipif(REPOSITORY_ROOT is None, reason="coverage checker is absent from the wheel")
def test_coverage_scope_checker_fails_closed(tmp_path: Path) -> None:
    """The stable-scope checker accepts its floor and rejects a regression."""
    assert REPOSITORY_ROOT is not None
    checker = REPOSITORY_ROOT / "scripts" / "check_coverage_scopes.py"
    config = tmp_path / "scopes.json"
    config.write_text(
        json.dumps(
            {
                "whole": {"minimum_percent": 50.0},
                "stable": {
                    "minimum_percent": 80.0,
                    "exclude_path_fragments": ["/experimental/"],
                },
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "coverage.json"

    def write_report(stable_covered: int) -> None:
        payload = {
            "totals": {"covered_lines": stable_covered, "num_statements": 20},
            "files": {
                "src/stochvolmodels/stable.py": {
                    "summary": {"covered_lines": stable_covered, "num_statements": 10}
                },
                "src/stochvolmodels/experimental/work.py": {
                    "summary": {"covered_lines": 0, "num_statements": 10}
                },
            },
        }
        report.write_text(json.dumps(payload), encoding="utf-8")

    write_report(stable_covered=10)
    accepted = subprocess.run(
        [sys.executable, str(checker), str(report), "--config", str(config)],
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "stable coverage: 100.00%" in accepted.stdout

    write_report(stable_covered=7)
    rejected = subprocess.run(
        [sys.executable, str(checker), str(report), "--config", str(config)],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "coverage ratchet failed" in rejected.stderr


@pytest.mark.repository_only
@pytest.mark.skipif(REPOSITORY_ROOT is None, reason="boundary checker is absent from the wheel")
def test_dependency_boundary_checker_rejects_optional_leaks(tmp_path: Path) -> None:
    """Fresh core/dev requirement trees cannot acquire an optional provider silently."""
    assert REPOSITORY_ROOT is not None
    checker = REPOSITORY_ROOT / "scripts" / "check_dependency_boundaries.py"
    core = tmp_path / "core.txt"
    dev = tmp_path / "dev.txt"
    core.write_text("numpy==2.3.0\n", encoding="utf-8")
    dev.write_text("numpy==2.3.0\npytest==9.0.0\n", encoding="utf-8")
    command = [
        sys.executable,
        str(checker),
        str(core),
        str(dev),
        "--pyproject",
        str(REPOSITORY_ROOT / "pyproject.toml"),
    ]
    accepted = subprocess.run(command, capture_output=True, text=True)
    assert accepted.returncode == 0, accepted.stderr

    dev.write_text("numpy==2.3.0\nqis==3.5.7\n", encoding="utf-8")
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "optional distributions leaked" in rejected.stderr
