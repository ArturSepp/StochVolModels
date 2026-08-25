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
    required_runtime_files = {
        "stochvolmodels/__init__.py",
        "stochvolmodels/data/model_paths.py",
        "stochvolmodels/models/__init__.py",
        "stochvolmodels/models/logsv.py",
        "stochvolmodels/models/regime_logsv.py",
        "stochvolmodels/models/regime_logsv_simulation.py",
        "stochvolmodels/models/tgarch.py",
        "stochvolmodels/pricers/regime_switch_logsv_pricer.py",
        "stochvolmodels/products/__init__.py",
        "stochvolmodels/products/payoffs.py",
        "stochvolmodels/valuation.py",
    }
    missing_runtime_files = required_runtime_files.difference(members)
    assert not missing_runtime_files, (
        f"required runtime files missing from wheel: {sorted(missing_runtime_files)}"
    )
    test_modules = {
        member
        for member in members
        if member.startswith("stochvolmodels/tests/test_") and member.endswith(".py")
    }
    assert len(test_modules) == 31, (
        f"expected exactly 31 automated test modules, found {len(test_modules)}"
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
