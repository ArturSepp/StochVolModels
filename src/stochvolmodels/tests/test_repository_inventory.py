"""Regression checks for public file inventories beside ignored private research.

Exercise the actual paper and dispatcher contracts in isolated repositories and
source archives, without reading or changing the maintainer's private files.
"""

import os
from pathlib import Path
import shutil
import subprocess
from typing import Callable

import pytest

from stochvolmodels.tests import test_joss_evidence as joss
from stochvolmodels.tests import test_module_layout as layout


pytestmark = [
    pytest.mark.repository_only,
    pytest.mark.skipif(
        layout.REPOSITORY_ROOT is None or not joss.PAPERS_README_PATH.is_file(),
        reason="repository inventory contracts are absent from installed wheels",
    ),
]


@pytest.fixture
def git_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an isolated Git index without commits or user configuration."""
    if shutil.which("git") is None:
        pytest.skip("Git is required for checkout inventory checks")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch=main", str(tmp_path)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return tmp_path


def _public_paper(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind a synthetic one-paper classification table to the real contract."""
    paper = root / "papers" / "public paper Ω"
    paper.mkdir(parents=True)
    (paper / "notes.md").write_text("Synthetic public paper.\n", encoding="utf-8")
    readme = root / "papers" / "README.md"
    readme.write_text(
        "## Papers\n| `public paper Ω` | synthetic |\n## Running\n", encoding="utf-8",
    )
    monkeypatch.setattr(joss, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(joss, "PAPERS_README_PATH", readme)
    monkeypatch.setattr(layout, "REPOSITORY_ROOT", root)


def _candidate(root: Path, visibility: str, content: str) -> None:
    """Add either an ignored private file or a public inventory violation."""
    path = root / "papers" / "draft Ω" / "legacy script.py"
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    if visibility != "untracked":
        (root / ".gitignore").write_text("/papers/draft Ω/\n", encoding="utf-8")
    if visibility == "tracked_ignored":
        subprocess.run(
            ["git", "add", "--force", "--", "papers/draft Ω/legacy script.py"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=30,
        )


@pytest.mark.parametrize("visibility", ["ignored", "untracked", "tracked_ignored"])
def test_paper_inventory_preserves_public_classification(
    git_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    repository_files: Callable[[Path, str], list[Path]],
    visibility: str,
) -> None:
    """Only ignored, untracked private papers may lack a classification row."""
    _public_paper(git_repository, monkeypatch)
    _candidate(git_repository, visibility, "Private synthetic note.\n")
    if visibility == "ignored":
        joss.test_every_paper_directory_has_exactly_one_status_row(repository_files)
    else:
        with pytest.raises(AssertionError):
            joss.test_every_paper_directory_has_exactly_one_status_row(repository_files)


@pytest.mark.parametrize("visibility", ["ignored", "untracked", "tracked_ignored"])
def test_dispatcher_inventory_preserves_public_violations(
    git_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    repository_files: Callable[[Path, str], list[Path]],
    visibility: str,
) -> None:
    """Do not parse ignored private code, but reject public legacy dispatchers."""
    _public_paper(git_repository, monkeypatch)
    content = "invalid private Python ?\n" if visibility == "ignored" else (
        "def run_local_test():\n    pass\n"
    )
    _candidate(git_repository, visibility, content)
    if visibility == "ignored":
        layout.test_repository_plural_dispatchers_use_current_api(repository_files)
    else:
        with pytest.raises(AssertionError, match="retains the old plural dispatcher"):
            layout.test_repository_plural_dispatchers_use_current_api(repository_files)


def test_archive_inventory_checks_included_files(
    git_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    repository_files: Callable[[Path, str], list[Path]],
) -> None:
    """Archives use their own files even when extracted beneath another Git checkout."""
    def unexpected_git(*args: object, **kwargs: object) -> None:
        pytest.fail("source archives must not invoke Git")

    monkeypatch.setattr(subprocess, "run", unexpected_git)
    archive = git_repository / "source archive"
    _public_paper(archive, monkeypatch)
    joss.test_every_paper_directory_has_exactly_one_status_row(repository_files)
    layout.test_repository_plural_dispatchers_use_current_api(repository_files)
    _candidate(archive, "untracked", "def run_local_test():\n    pass\n")
    with pytest.raises(AssertionError):
        joss.test_every_paper_directory_has_exactly_one_status_row(repository_files)
    with pytest.raises(AssertionError, match="retains the old plural dispatcher"):
        layout.test_repository_plural_dispatchers_use_current_api(repository_files)


def test_git_inventory_failure_does_not_scan_private_files(
    git_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    repository_files: Callable[[Path, str], list[Path]],
) -> None:
    """A checkout's Git failure must not fall back to scanning private content."""
    def fail_git(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, ["git", "ls-files"])

    monkeypatch.setattr(subprocess, "run", fail_git)
    with pytest.raises(subprocess.CalledProcessError):
        repository_files(git_repository, "papers")
