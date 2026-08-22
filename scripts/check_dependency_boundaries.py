"""Reject optional distributions in freshly resolved core and development trees."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

MODULE_DISTRIBUTIONS = {
    "option_chain_analytics": {"option-chain-analytics"},
    "sklearn": {"scikit-learn"},
    "yaml": {"pyyaml"},
}
# pytest-regressions legitimately brings PyYAML into the contributor-only dev tree. It remains
# forbidden from core and from module-level first-party imports.
TREE_ALLOWANCES = {"dev": {"pyyaml"}}


def _normalise(name: str) -> str:
    """Return the PEP 503-normalized spelling of a distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _resolved(path: Path) -> set[str]:
    """Extract normalized distribution names from a compiled requirements file."""
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", maxsplit=1)[0].strip()
        if requirement and not requirement.startswith(("-", ";")):
            names.add(_normalise(re.split(r"[<>=!~\[ ]", requirement, maxsplit=1)[0]))
    return names


def check_boundaries(pyproject: Path, requirement_files: list[Path]) -> None:
    """Assert optional module providers are absent from every requirements tree."""
    with pyproject.open("rb") as stream:
        config = tomllib.load(stream)
    modules = config["tool"]["ruff"]["lint"]["flake8-tidy-imports"][
        "banned-module-level-imports"
    ]
    assert modules, "optional-module boundary list is empty"

    failures = []
    for requirements in requirement_files:
        resolved = _resolved(requirements)
        allowed = TREE_ALLOWANCES.get(requirements.stem.lower(), set())
        leaks = []
        for module in modules:
            providers = MODULE_DISTRIBUTIONS.get(module, {_normalise(module)})
            leaks.extend(
                provider for provider in providers if provider in resolved and provider not in allowed
            )
        leaks = sorted(set(leaks))
        if leaks:
            failures.append(f"{requirements.name}: {', '.join(leaks)}")
        else:
            print(f"{requirements.name}: clean ({len(resolved)} distributions)")
    if failures:
        raise AssertionError("optional distributions leaked into core/dev: " + "; ".join(failures))


def main() -> None:
    """Parse requirement trees and enforce the package boundary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("requirements", nargs="+", type=Path)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    args = parser.parse_args()
    check_boundaries(args.pyproject, args.requirements)


if __name__ == "__main__":
    main()
