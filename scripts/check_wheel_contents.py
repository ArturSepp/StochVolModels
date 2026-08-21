"""Validate that a StochVolModels wheel contains only intentional runtime/test files."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile


def check_wheel(wheel_path: Path) -> None:
    """Raise ``AssertionError`` when *wheel_path* violates the artifact contract."""
    with ZipFile(wheel_path) as wheel:
        members = tuple(wheel.namelist())

    unexpected_roots = {
        member.split("/", 1)[0]
        for member in members
        if not (
            member.startswith("stochvolmodels/")
            or (
                member.split("/", 1)[0].startswith("stochvolmodels-")
                and member.split("/", 1)[0].endswith(".dist-info")
            )
        )
    }
    assert not unexpected_roots, f"unexpected wheel roots: {sorted(unexpected_roots)}"

    forbidden_prefixes = (
        "stochvolmodels/examples/",
        "examples/",
        "papers/",
        "docs/",
        "resources/",
        "stochvolmodels/pde_solvers/",
    )
    forbidden = [member for member in members if member.startswith(forbidden_prefixes)]
    assert not forbidden, f"repository-only files entered wheel: {forbidden[:10]}"
    assert "stochvolmodels/settings.yaml" not in members, (
        "machine-local settings.yaml entered the wheel"
    )
    assert "stochvolmodels/__init__.py" in members
    assert any(member.startswith("stochvolmodels/tests/test_") for member in members)
    assert any(member.endswith(".npz") for member in members), "regression baseline is missing"
    assert not any(member.endswith((".pyc", ".pyo")) for member in members)

    print(f"wheel-content-check: PASS ({len(members)} files): {wheel_path.name}")


def main() -> None:
    """Parse a wheel path and validate it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    check_wheel(args.wheel)


if __name__ == "__main__":
    main()
