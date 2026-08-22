"""Reject optional distributions in freshly resolved core and development trees."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI lane
    tomllib = None

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


def _banned_modules(pyproject: Path) -> list[str]:
    """Load Ruff's banned module list, including on Python 3.10 without dependencies."""
    if tomllib is not None:
        with pyproject.open("rb") as stream:
            config = tomllib.load(stream)
        return config["tool"]["ruff"]["lint"]["flake8-tidy-imports"][
            "banned-module-level-imports"
        ]

    text = pyproject.read_text(encoding="utf-8")
    section = re.search(
        r"^\[tool\.ruff\.lint\.flake8-tidy-imports\]\s*$([\s\S]*?)(?=^\[|\Z)",
        text,
        flags=re.MULTILINE,
    )
    if section is None:
        raise AssertionError("Ruff flake8-tidy-imports section is missing")
    assignment = re.search(
        r"banned-module-level-imports\s*=\s*\[([\s\S]*?)\]",
        section.group(1),
    )
    if assignment is None:
        raise AssertionError("optional-module boundary list is missing")
    return re.findall(r'["\']([^"\']+)["\']', assignment.group(1))


def check_boundaries(pyproject: Path, requirement_files: list[Path]) -> None:
    """Assert optional module providers are absent from every requirements tree."""
    modules = _banned_modules(pyproject)
    assert modules, "optional-module boundary list is empty"

    failures = []
    for requirements in requirement_files:
        resolved = _resolved(requirements)
        allowed = TREE_ALLOWANCES.get(requirements.stem.lower(), set())
        leaks = []
        for module in modules:
            providers = MODULE_DISTRIBUTIONS.get(module, {_normalise(module)})
            leaks.extend(
                provider
                for provider in providers
                if provider in resolved and provider not in allowed
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
