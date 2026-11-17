"""Package-local pytest configuration and repository file inventory fixtures."""

import os
import subprocess
from pathlib import Path
from typing import Callable

import pytest


def pytest_configure(config) -> None:
    """Register package-local markers outside the source checkout."""
    markers = (
        "pde_reference: compare with the repository's historical LSV implementation",
        "slow: numerical regression or simulation tests excluded from the fast CI lane",
        "optional_integration: provider or optional-extra contract that skips on a core install",
        "repository_only: documentation, example, or paper contract absent from the wheel",
        "paper_replication: long repository paper-reproduction gate run outside the core suite",
    )
    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(items) -> None:
    """The independent LSV comparison lives alongside the source checkout."""
    root = Path(__file__).resolve().parents[3]
    if not (root / "scripts/check_wheel_contents.py").is_file():
        missing = pytest.mark.skip(reason="historical LSV comparisons require the source checkout")
        for item in items:
            if item.get_closest_marker("pde_reference"):
                item.add_marker(missing)


@pytest.fixture
def repository_files() -> Callable[[Path, str], list[Path]]:
    """List Git-visible files, or included archive files when Git metadata is absent."""
    def collect(root: Path, directory: str) -> list[Path]:
        """Keep tracked and nonignored untracked files in the requested directory."""
        if (root / ".git").exists():
            completed = subprocess.run(
                [
                    "git", "ls-files", "-z", "--cached", "--others",
                    "--exclude-standard", "--", directory,
                ],
                cwd=root,
                capture_output=True,
                check=True,
                timeout=30,
            )
            paths = [root / os.fsdecode(name) for name in completed.stdout.split(b"\0") if name]
        else:
            paths = list((root / directory).rglob("*"))
        return sorted({path for path in paths if path.is_file()})

    return collect


@pytest.fixture
def package_modules(repository_files) -> list[Path]:
    """Inspect visible source modules in a checkout and actual files in an installed wheel."""
    package = Path(__file__).resolve().parents[1]
    root = package.parents[1]
    if package.parent.name == "src" and (root / "pyproject.toml").is_file():
        return [p for p in repository_files(root, "src/stochvolmodels") if p.suffix == ".py"]
    return sorted(package.rglob("*.py"))
