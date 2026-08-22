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
    development = [
        member for member in members if "/run_local/" in member or member.endswith("_run.py")
    ]
    assert not development, f"development runners entered wheel: {development[:10]}"
    assert "stochvolmodels/settings.yaml" not in members, (
        "machine-local settings.yaml entered the wheel"
    )
    assert "stochvolmodels/__init__.py" in members
    test_modules = {
        member
        for member in members
        if member.startswith("stochvolmodels/tests/test_") and member.endswith(".py")
    }
    assert len(test_modules) == 21, (
        f"expected exactly 21 automated test modules, found {len(test_modules)}"
    )
    assert not any(member.endswith("_test.py") for member in members), (
        "automated tests must use the test_*.py form"
    )
    baselines = {member for member in members if member.endswith(".npz")}
    assert baselines == {
        "stochvolmodels/tests/test_rough_logsv_pricer_regression/"
        "test_rough_logsv_pricer_pricing_regression.npz"
    }, f"unexpected regression baselines: {sorted(baselines)}"
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
