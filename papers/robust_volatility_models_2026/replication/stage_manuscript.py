"""Synchronize or verify the review's embedded LaTeX and figure files.

Run --sync after regenerating exhibits in replication/. The default --check
compares the embedded blocks and figures with their replication sources.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

MODULE = Path(__file__).resolve().parent.parent
CODE = MODULE / "replication"
PAPER = MODULE / "paper"
MANUSCRIPT = PAPER / "robust_volatility_models.tex"

SECTIONS = {
    name: f"empirical_bloomberg_20261030/{name}"
    for name in ("empirical_macros.tex", "empirical_section.tex")
}
SECTIONS.update({name: f"calibration_section_20261030/{name}"
                 for name in ("calibration_macros.tex", "calibration_section.tex")})
SECTIONS["physical_pricing.tex"] = "physical_pricing_section_20261127/physical_pricing.tex"
for name in ("quarterly_premia.tex", "quarterly_macros.tex"):
    SECTIONS[name] = f"cross_asset_physical_pricing_20261124/quarterly_panel/{name}"

BLOCK = re.compile(
    r"^(?P<start>\\ReplicationBlock\{begin\}\{(?P<name>[\w.-]+\.tex)\}\n)"
    r"(?P<body>.*?)"
    r"(?P<end>^\\ReplicationBlock\{end\}\{(?P=name)\}$)",
    re.MULTILINE | re.DOTALL,
)
MARKER = re.compile(r"^\\ReplicationBlock\b", re.MULTILINE)
GRAPHIC = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\s*\{)([^}]+)(\})")
FILE_COMMENT = re.compile(r"^[ \t]*%[^\n]*\n?", re.MULTILINE)
COMMENT = re.compile(r"(?<!\\)((?:\\\\)*)%[^\n]*")
TEX_INPUT = re.compile(r"\\(?:input|include)\b")


def render_block(source: Path) -> str:
    """Omit standalone comments and point graphics at their manuscript copies."""
    text = FILE_COMMENT.sub("", source.read_text(encoding="utf-8-sig"))

    def rewrite_graphic(match):
        parts = (part for part in Path(match[2]).parts if part != "figures")
        return match[1] + "figures/" + "/".join(parts) + match[3]

    text = GRAPHIC.sub(rewrite_graphic, text)
    return text if text.endswith("\n") or not text else text + "\n"


def figure_source(relative: str) -> tuple[Path, Path]:
    path = Path(relative)
    destination = (PAPER / path).resolve()
    if not destination.is_relative_to((PAPER / "figures").resolve()) or path.parts[:1] != ("figures",):
        raise ValueError(f"Graphic must be in paper/figures/: {relative}")
    original = Path(*path.parts[1:])
    if not original.parts or ".." in original.parts:
        raise ValueError(f"Invalid figure path: {relative}")
    candidates = (CODE / original, CODE / original.parts[0] / "figures" / Path(*original.parts[1:]))
    sources = [path for path in candidates if path.is_file()]
    if len(sources) != 1:
        raise ValueError(f"Expected one source for {relative}; found {sources}")
    return sources[0], destination


def exhibit_record(source: Path, destination: Path, content: bytes) -> dict[str, str]:
    return {
        "source": source.relative_to(MODULE).as_posix(),
        "destination": destination.relative_to(MODULE).as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "staged_sha256": hashlib.sha256(content).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sync", action="store_true", help="refresh generated TeX blocks and figures")
    mode.add_argument("--check", action="store_true", help="check only (default)")
    args = parser.parse_args()
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    if 2 * len(list(BLOCK.finditer(manuscript))) != len(MARKER.findall(manuscript)):
        raise ValueError("The manuscript contains an unmatched or malformed replication block marker")
    records = {}

    def refresh_block(match):
        name = match["name"]
        if name in records or MARKER.search(match["body"]):
            raise ValueError(f"Duplicate or nested replication block: {name}")
        if name not in SECTIONS:
            raise ValueError(f"Unregistered replication block: {name}")
        source = CODE / SECTIONS[name]
        expected = render_block(source)
        if not args.sync and match["body"] != expected:
            raise RuntimeError(f"Stale manuscript block: {name}; run --sync")
        record = exhibit_record(source, MANUSCRIPT, expected.encode("utf-8"))
        record["block"] = name
        records[name] = record
        return match["start"] + expected + match["end"]

    updated = BLOCK.sub(refresh_block, manuscript)
    active = COMMENT.sub(r"\1", updated)
    if TEX_INPUT.search(active):
        raise ValueError("Inline external TeX inputs in the manuscript before staging")
    pending = {}
    for relative in dict.fromkeys(match[2] for match in GRAPHIC.finditer(active)):
        source, destination = figure_source(relative)
        expected = source.read_bytes()
        if not destination.is_file() or destination.read_bytes() != expected:
            if not args.sync:
                raise RuntimeError(f"Missing or stale exhibit: {relative}; run --sync")
            pending[destination] = expected
        records[relative] = exhibit_record(source, destination, expected)
    if args.sync:
        for destination, content in pending.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        if updated != manuscript:
            MANUSCRIPT.write_text(updated, encoding="utf-8", newline="\n")
        manifest = MODULE.parents[1] / "agents" / "robust_volatility_review_20261030" / "exhibit_manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        record = {"main_files": [MANUSCRIPT.name], "dependency_count": len(records), "files": records}
        manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mode": "sync" if args.sync else "check",
                      "main_files": 1, "dependencies_verified": len(records)}))


if __name__ == "__main__":
    main()
