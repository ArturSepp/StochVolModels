"""Validate that a StochVolModels sdist excludes repository-only material."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from check_wheel_contents import REQUIRED_RUNTIME_FILES


def check_sdist(sdist_path: Path) -> None:
    """Raise ``AssertionError`` when *sdist_path* violates the artifact contract."""
    with tarfile.open(sdist_path, mode="r:*") as archive:
        members = tuple(member.name.rstrip("/") for member in archive.getmembers())

    roots = {member.split("/", 1)[0] for member in members if member}
    assert len(roots) == 1, f"expected one sdist root, found: {sorted(roots)}"
    root = roots.pop()
    assert root.startswith("stochvolmodels-"), f"unexpected sdist root: {root}"

    prefix = f"{root}/"
    payload = tuple(member.removeprefix(prefix) for member in members if member != root)
    assert all(member and not member.startswith(("/", "../")) for member in payload), (
        "sdist contains an absolute, empty, or parent-traversing member"
    )

    forbidden_prefixes = (
        "agents/",
        "docs/",
        "examples/",
        "outputs/",
        "papers/",
        "resources/",
        "volatility_book/",
        "src/stochvolmodels/pde_solvers/",
    )
    forbidden = [member for member in payload if member.startswith(forbidden_prefixes)]
    assert not forbidden, f"repository-only files entered sdist: {forbidden[:10]}"

    development = [
        member for member in payload if "/run_local/" in member or member.endswith("_run.py")
    ]
    assert not development, f"development runners entered sdist: {development[:10]}"
    assert "src/stochvolmodels/settings.yaml" not in payload, (
        "machine-local settings.yaml entered the sdist"
    )

    required = {
        "CHANGELOG.md",
        "LICENSE.txt",
        "MANIFEST.in",
        "README.md",
        "pyproject.toml",
        "src/stochvolmodels/settings.yaml.example",
        *(f"src/{member}" for member in REQUIRED_RUNTIME_FILES),
    }
    missing = required.difference(payload)
    assert not missing, f"required sdist files missing: {sorted(missing)}"

    test_modules = {
        member
        for member in payload
        if member.startswith("src/stochvolmodels/tests/test_") and member.endswith(".py")
    }
    assert len(test_modules) == 32, (
        f"expected exactly 32 automated test modules, found {len(test_modules)}"
    )
    baselines = {member for member in payload if member.endswith(".npz")}
    assert baselines == {
        "src/stochvolmodels/tests/test_rough_logsv_pricer_regression/"
        "test_rough_logsv_pricer_pricing_regression.npz"
    }, f"unexpected regression baselines: {sorted(baselines)}"
    assert not any(member.endswith((".pyc", ".pyo")) for member in payload)

    print(f"sdist-content-check: PASS ({len(members)} members): {sdist_path.name}")


def main() -> None:
    """Parse an sdist path and validate it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()
    check_sdist(args.sdist)


if __name__ == "__main__":
    main()
