"""Package-local pytest configuration and repository file inventory fixtures."""

import os
import subprocess
from pathlib import Path
from typing import Callable

import pytest


def pytest_configure(config) -> None:
    """Register package-local markers outside the source checkout."""
    markers = (
        "slow: numerical regression or simulation tests excluded from the fast CI lane",
        "optional_integration: provider or optional-extra contract that skips on a core install",
        "repository_only: documentation, example, or paper contract absent from the wheel",
        "paper_replication: long repository paper-reproduction gate run outside the core suite",
    )
    for marker in markers:
        config.addinivalue_line("markers", marker)


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
