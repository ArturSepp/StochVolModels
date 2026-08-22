"""Validate the JOSS manuscript using only the Python standard library."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REQUIRED_HEADINGS = (
    "# Summary",
    "# Statement of need",
    "# State of the field",
    "# Software design",
    "# Research impact statement",
    "# AI usage disclosure",
    "# Acknowledgements",
    "# References",
)


def _front_matter(text: str) -> tuple[str, str]:
    """Return YAML-like front matter and manuscript body, rejecting malformed delimiters."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise AssertionError("paper.md must start with YAML front matter")
    try:
        end = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as error:
        raise AssertionError("paper.md has no closing front-matter delimiter") from error
    return "\n".join(lines[1:end]), "\n".join(lines[end + 1 :])


def validate(paper_path: Path, bibliography_path: Path) -> None:
    """Raise ``AssertionError`` when the manuscript violates the submission contract."""
    paper = paper_path.read_text(encoding="utf-8")
    bibliography = bibliography_path.read_text(encoding="utf-8")
    front, body = _front_matter(paper)

    for field in ("title:", "authors:", "affiliations:", "bibliography:"):
        assert field in front, f"missing front-matter field: {field}"
    assert "bibliography: paper.bib" in front, "paper must reference paper.bib"

    missing = [heading for heading in REQUIRED_HEADINGS if heading not in body]
    assert not missing, f"missing required headings: {missing}"
    words = len(body.split())
    assert 750 <= words <= 1750, f"paper body has {words} words; expected 750..1750"
    assert re.search(r"\b(TODO|TBD|FIXME|XXX)\b", paper, re.IGNORECASE) is None, (
        "paper contains an unresolved placeholder"
    )

    citations = set(re.findall(r"@([A-Za-z0-9_:-]+)", body))
    entries = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bibliography))
    assert citations, "paper contains no citations"
    assert citations <= entries, f"unknown citation keys: {sorted(citations - entries)}"

    author_affiliations = {
        int(value) for value in re.findall(r"^\s+affiliation:\s*(\d+)\s*$", front, re.MULTILINE)
    }
    affiliation_indices = {
        int(value) for value in re.findall(r"^\s+index:\s*(\d+)\s*$", front, re.MULTILINE)
    }
    assert author_affiliations, "authors have no affiliation references"
    assert author_affiliations <= affiliation_indices, "author affiliation index is undefined"
    print(f"joss-paper-check: PASS ({words} words, {len(citations)} citations)")


def main() -> None:
    """Parse manuscript paths and run the validator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, default=Path("paper.md"))
    parser.add_argument("--bibliography", type=Path, default=Path("paper.bib"))
    args = parser.parse_args()
    validate(args.paper, args.bibliography)


if __name__ == "__main__":
    main()
