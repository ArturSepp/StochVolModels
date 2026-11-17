"""Validate that a StochVolModels wheel contains only intentional runtime/test files."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile


SOURCE_TEST_ROOT = Path(__file__).resolve().parents[1] / "src/stochvolmodels/tests"


# Exact distribution contract for the reviewed PDE foundation.
PDE_FILES = {
    "stochvolmodels/pde_solvers/__init__.py",
    "stochvolmodels/pde_solvers/adi.py",
    "stochvolmodels/pde_solvers/boundaries.py",
    "stochvolmodels/pde_solvers/coefficients.py",
    "stochvolmodels/pde_solvers/diagnostics.py",
    "stochvolmodels/pde_solvers/grids.py",
    "stochvolmodels/pde_solvers/jumps.py",
    "stochvolmodels/pde_solvers/operators_1d.py",
    "stochvolmodels/pde_solvers/operators_2d.py",
    "stochvolmodels/pde_solvers/poisson.py",
    "stochvolmodels/pde_solvers/solver_1d.py",
    "stochvolmodels/pde_solvers/solver_2d.py",
    "stochvolmodels/pde_solvers/tridiagonal.py",
}


def check_pde_files(members) -> None:
    """Reject missing modules and checkout-only PDE material in either distribution."""
    actual = {name for name in members if name.startswith("stochvolmodels/pde_solvers/")}
    assert actual == PDE_FILES, (
        f"PDE file inventory differs: missing={sorted(PDE_FILES - actual)}, "
        f"unexpected={sorted(actual - PDE_FILES)}"
    )


REQUIRED_RUNTIME_FILES = PDE_FILES | {
    "stochvolmodels/__init__.py",
    "stochvolmodels/data/model_paths.py",
    "stochvolmodels/models/__init__.py",
    "stochvolmodels/models/inverse_gamma_normal.py",
    "stochvolmodels/models/logsv.py",
    "stochvolmodels/models/logsv_pde.py",
    "stochvolmodels/fitters/dupire.py",
    "stochvolmodels/fitters/lsv_fixed_point.py",
    "stochvolmodels/fitters/lsv_parametric.py",
    "stochvolmodels/fitters/lsv_market.py",
    "stochvolmodels/fitters/lsv_market_pipeline.py",
    "stochvolmodels/models/local_vol.py",
    "stochvolmodels/models/lsv.py",
    "stochvolmodels/models/lsv_simulation.py",
    "stochvolmodels/pricers/density_pricer.py",
    "stochvolmodels/pricers/lsv_mc_pricer.py",
    "stochvolmodels/pricers/lsv_projection.py",
    "stochvolmodels/pricers/local_vol_pde_pricer.py",
    "stochvolmodels/pricers/logsv_pde_pricer.py",
    "stochvolmodels/models/regime_logsv.py",
    "stochvolmodels/models/regime_logsv_simulation.py",
    "stochvolmodels/models/tgarch.py",
    "stochvolmodels/pricers/regime_switch_logsv_pricer.py",
    "stochvolmodels/products/__init__.py",
    "stochvolmodels/products/payoffs.py",
    "stochvolmodels/valuation.py",
}


def check_wheel(wheel_path: Path) -> None:
    """Raise ``AssertionError`` when *wheel_path* violates the artifact contract."""
    with ZipFile(wheel_path) as wheel:
        members = tuple(wheel.namelist())

    check_pde_files(members)
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
        "volatility_book/",
        "docs/",
        "resources/",
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
    missing_runtime_files = REQUIRED_RUNTIME_FILES.difference(members)
    assert not missing_runtime_files, (
        f"required runtime files missing from wheel: {sorted(missing_runtime_files)}"
    )
    test_modules = {
        member
        for member in members
        if member.startswith("stochvolmodels/tests/test_") and member.endswith(".py")
    }
    expected_test_modules = {
        f"stochvolmodels/tests/{path.name}" for path in SOURCE_TEST_ROOT.glob("test_*.py")
    }
    assert expected_test_modules, "source test module inventory is empty"
    assert test_modules == expected_test_modules, (
        "wheel test module inventory differs from source: "
        f"missing={sorted(expected_test_modules - test_modules)}, "
        f"unexpected={sorted(test_modules - expected_test_modules)}"
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
