"""Stage the review's generated TeX and figures, or verify their current copies.

Run --sync after regenerating an experiment's exhibits. The default --check
verifies dependencies without changing the manuscript or running experiments.
Frozen experiment files and their original provenance manifests stay intact.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

MODULE = Path(__file__).resolve().parent.parent
CODE = MODULE / "replication"
PAPER = MODULE / "paper"
MAINS = ("robust_volatility_models.tex",)
SECTIONS = {
    name: f"empirical_bloomberg_20261030/{name}"
    for name in ("empirical_macros.tex", "empirical_section.tex", "empirical_supplement.tex")
}
SECTIONS.update({name: f"calibration_section_20261030/{name}"
                 for name in ("calibration_macros.tex", "calibration_section.tex")})
REFERENCE = re.compile(r"(\\(?:input|include|includegraphics)(?:\[[^\]]*\])?\s*\{)([^}]+)(\})")


def rewrite_dependencies(text):
    """Change paths only in generated TeX; numerical/prose content stays identical."""
    def replacement(match):
        name = match[2]
        if "includegraphics" in match[1]:
            name = "figures/" + "/".join(p for p in Path(name).parts if p != "figures")
        else:
            name = "sections/" + Path(name).with_suffix(".tex").name
        return match[1] + name + match[3]
    return REFERENCE.sub(replacement, text)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sync", action="store_true", help="stage generated dependencies")
    mode.add_argument("--check", action="store_true", help="check only (default)")
    args = parser.parse_args()
    records, visited = {}, set()

    def stage(relative, is_tex):
        destination = (PAPER / relative).resolve()
        if not destination.is_relative_to(PAPER.resolve()):
            raise ValueError(f"Dependency outside paper folder: {relative}")
        if is_tex:
            if Path(relative).parent.as_posix() != "sections":
                raise ValueError(f"Generated TeX must be in sections/: {relative}")
            source = CODE / SECTIONS[Path(relative).name]
            expected = rewrite_dependencies(source.read_text(encoding="utf-8-sig")).encode("utf-8")
        else:
            path = Path(relative)
            if path.parts[0] != "figures":
                raise ValueError(f"Graphics must be in figures/: {relative}")
            original = Path(*path.parts[1:])
            candidates = [CODE / original,
                          CODE / original.parts[0] / "figures" / Path(*original.parts[1:])]
            candidates = [p for p in candidates if p.is_file()]
            if len(candidates) != 1:
                raise ValueError(f"Expected one source for {relative}; found {candidates}")
            source = candidates[0]
            expected = source.read_bytes()
        if args.sync:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(expected)
        if not destination.is_file() or destination.read_bytes() != expected:
            raise RuntimeError(f"Missing or stale exhibit: {relative}; run --sync")
        records[relative] = {"source": source.relative_to(MODULE).as_posix(),
                         "destination": destination.relative_to(MODULE).as_posix(),
                         "source_sha256": digest(source), "staged_sha256": digest(destination)}
        return destination

    def visit(path):
        if path in visited:
            return
        visited.add(path)
        text = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text(encoding="utf-8-sig"))
        for relative in re.findall(r"\\(?:input|include)\s*\{([^}]+)\}", text):
            visit(stage(relative, True))
        for relative in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}", text):
            stage(relative, False)

    for name in MAINS:
        visit(PAPER / name)
    record = {"main_files": list(MAINS), "dependency_count": len(records), "files": records}
    if args.sync:
        audit = MODULE.parents[1] / "agents" / "robust_volatility_review_20261030" / "exhibit_manifest.json"
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mode": "sync" if args.sync else "check",
                      "main_files": len(MAINS), "dependencies_verified": len(records)}))


if __name__ == "__main__":
    main()
