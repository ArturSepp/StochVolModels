"""Repository-only contracts for the JOSS manuscript and its measured evidence."""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import stochvolmodels

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PAPER_PATH = REPOSITORY_ROOT / "paper.md"
EVIDENCE_PATH = REPOSITORY_ROOT / "docs" / "audit" / "joss_evidence.json"
API_PATH = REPOSITORY_ROOT / "docs" / "api.md"
BIB_PATH = REPOSITORY_ROOT / "paper.bib"
EXAMPLES_README_PATH = REPOSITORY_ROOT / "examples" / "README.md"
PAPERS_README_PATH = REPOSITORY_ROOT / "papers" / "README.md"
AUTHORSHIP_PATH = REPOSITORY_ROOT / "docs" / "audit" / "joss_authorship.json"
AI_USE_PATH = REPOSITORY_ROOT / "docs" / "audit" / "ai_use.json"
RESEARCH_IMPACT_PATH = REPOSITORY_ROOT / "docs" / "audit" / "research_impact.json"

pytestmark = [
    pytest.mark.repository_only,
    pytest.mark.skipif(
        not PAPER_PATH.is_file(),
        reason="JOSS repository artifacts are intentionally absent from the installed wheel",
    ),
]


def _paper_body(text: str) -> str:
    """Return manuscript content after its YAML front matter."""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                return "\n".join(lines[index + 1 :])
    return text


def _evidence() -> dict:
    """Load the tracked evidence record."""
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def test_joss_paper_has_required_sections_and_word_count() -> None:
    """The manuscript follows the current JOSS structure and body-length guidance."""
    paper = PAPER_PATH.read_text(encoding="utf-8")
    required = [
        "# Summary",
        "# Statement of need",
        "# State of the field",
        "# Software design",
        "# Research impact statement",
        "# AI usage disclosure",
        "# Acknowledgements",
        "# References",
    ]
    assert all(heading in paper for heading in required)
    words = len(_paper_body(paper).split())
    assert 750 <= words <= 1750, words


def test_joss_citations_have_bibliography_entries() -> None:
    """Every manuscript citation key resolves in paper.bib."""
    paper = PAPER_PATH.read_text(encoding="utf-8")
    bibliography = BIB_PATH.read_text(encoding="utf-8")
    citations = set(re.findall(r"@([A-Za-z0-9_:-]+)", paper))
    entries = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bibliography))
    assert citations <= entries, sorted(citations - entries)


def test_standalone_joss_validator_rejects_mutations(tmp_path: Path) -> None:
    """The standard-library paper gate fails closed on representative submission defects."""
    validator = REPOSITORY_ROOT / "scripts" / "check_joss_paper.py"
    original = PAPER_PATH.read_text(encoding="utf-8")
    bibliography = tmp_path / "paper.bib"
    bibliography.write_text(BIB_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    def run(mutated: str) -> subprocess.CompletedProcess[str]:
        paper = tmp_path / "paper.md"
        paper.write_text(mutated, encoding="utf-8")
        return subprocess.run(
            [
                sys.executable,
                str(validator),
                "--paper",
                str(paper),
                "--bibliography",
                str(bibliography),
            ],
            capture_output=True,
            text=True,
        )

    assert run(original).returncode == 0
    mutations = (
        original.replace("# Statement of need", "# Need", 1),
        original.replace("[@heston1993]", "[@unknown_reference]", 1),
        original + "\n" + "excess " * 1800,
        original.removeprefix("---\n"),
    )
    for mutated in mutations:
        completed = run(mutated)
        assert completed.returncode != 0, completed.stdout


def test_live_joss_evidence_matches_repository() -> None:
    """Stable-surface and manuscript measurements cannot drift silently."""
    evidence = _evidence()
    current = evidence["current"]
    metrics = current["metrics"]
    paper = PAPER_PATH.read_text(encoding="utf-8")
    api = API_PATH.read_text(encoding="utf-8")

    assert len(current["measured_at_commit"]) == 40
    assert metrics["stable_root_exports"]["value"] == len(stochvolmodels.__all__)
    assert metrics["stable_exports_rendered_in_api"]["value"] == sum(
        name in api for name in stochvolmodels.__all__
    )
    assert metrics["paper_body_words"]["value"] == len(_paper_body(paper).split())

    for metric in metrics.values():
        phrase = metric.get("paper_phrase")
        if phrase is not None:
            assert phrase in paper


def test_all_stable_exports_are_documented() -> None:
    """The stable public surface has Python docstrings and rendered API entries."""
    api = API_PATH.read_text(encoding="utf-8")
    missing_docstrings = [
        name
        for name in stochvolmodels.__all__
        if name != "__version__" and not inspect.getdoc(getattr(stochvolmodels, name))
    ]
    missing_api = [name for name in stochvolmodels.__all__ if name not in api]
    assert missing_docstrings == []
    assert missing_api == []


def test_every_root_example_has_a_reviewer_lane() -> None:
    """Repository examples cannot appear without an explicit data and automation classification."""
    examples_readme = EXAMPLES_README_PATH.read_text(encoding="utf-8")
    scripts = sorted(
        path.relative_to(REPOSITORY_ROOT / "examples").as_posix()
        for path in (REPOSITORY_ROOT / "examples").rglob("*.py")
        if path.name != "__init__.py"
    )
    lanes = examples_readme.split("## Lanes", maxsplit=1)[1].split("## Option data", maxsplit=1)[0]
    listed = re.findall(r"^\| `([^`]+\.py)` \|", lanes, re.MULTILINE)
    assert sorted(listed) == scripts
    assert len(listed) == len(set(listed)), "an example is classified more than once"


def test_every_paper_directory_has_exactly_one_status_row() -> None:
    """Published, development, and exploratory paper directories are all classified once."""
    papers_readme = PAPERS_README_PATH.read_text(encoding="utf-8")
    table = papers_readme.split("## Papers", maxsplit=1)[1].split("## Running", maxsplit=1)[0]
    listed = re.findall(r"^\| `([^`]+)` \|", table, re.MULTILINE)
    directories = sorted(
        path.name
        for path in (REPOSITORY_ROOT / "papers").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    )
    assert sorted(listed) == directories
    assert len(listed) == len(set(listed)), "a paper directory is classified more than once"


def test_human_evidence_ledgers_are_explicit_and_structured() -> None:
    """Authorship and AI decisions remain visible rather than being inferred by automation."""
    authorship = json.loads(AUTHORSHIP_PATH.read_text(encoding="utf-8"))
    ai_use = json.loads(AI_USE_PATH.read_text(encoding="utf-8"))
    impact = json.loads(RESEARCH_IMPACT_PATH.read_text(encoding="utf-8"))

    assert authorship["status"] == "approved_by_maintainer"
    assert authorship["proposed_authors"][0]["orcid"] == "0000-0002-7038-1748"
    assert ai_use["status"] == "approved_by_maintainer"
    assert {entry["provider"] for entry in ai_use["entries"]} == {"Anthropic", "OpenAI"}
    assert all(entry["model_or_version"] for entry in ai_use["entries"])
    assert any(record["joss_role"].startswith("primary") for record in impact["records"])
    assert impact["unmapped_exploratory_directories"]
