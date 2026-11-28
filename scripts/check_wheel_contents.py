"""Validate that a StochVolModels wheel contains only intentional runtime/test files."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile


SOURCE_TEST_ROOT = Path(__file__).resolve().parents[1] / "src/stochvolmodels/tests"

RETIRED_FORWARDERS = {
    "stochvolmodels/fitters/tdist.py",
    "stochvolmodels/models/logsv_pde.py",
    "stochvolmodels/models/lsv_simulation.py",
    "stochvolmodels/models/regime_logsv_simulation.py",
    "stochvolmodels/pricers/lsv_projection.py",
    "stochvolmodels/pricers/rough_logsv/_classical_pde.py",
    "stochvolmodels/pricers/rough_logsv/_constrained_kernel.py",
    "stochvolmodels/pricers/rough_logsv/_history.py",
    "stochvolmodels/pricers/rough_logsv/_history_fast.py",
    "stochvolmodels/pricers/rough_logsv/expm.py",
    "stochvolmodels/pricers/rough_logsv/history.py",
    "stochvolmodels/pricers/rough_logsv/kernel_fit.py",
    "stochvolmodels/pricers/rough_logsv/multilevel.py",
    "stochvolmodels/pricers/rough_logsv/params.py",
    "stochvolmodels/pricers/rough_logsv/rough_kernel.py",
}


PACKAGE_DATA_FILES = {
    "stochvolmodels/settings.yaml.example",
    "stochvolmodels/tests/test_rough_logsv_pricer_regression/"
    "test_rough_logsv_pricer_pricing_regression.npz",
}


def check_package_payload(members) -> None:
    """Check repository-only paths and the data shipped inside the import package."""
    files = [
        name for name in members if name.startswith("stochvolmodels/") and not name.endswith("/")
    ]
    guides = sorted(name for name in files if name.endswith("/README.md"))
    assert not guides, f"checkout documentation entered the package: {guides}"
    checkout_dirs = {
        "examples", "papers", "validation", "run_local", "volatility_book",
        "agents", "resources", "outputs", "docs",
    }
    development = [
        name for name in files
        if checkout_dirs.intersection(name.split("/")[1:-1])
        or name.rsplit("/", 1)[-1].startswith("run_")
        or name.endswith("_run.py")
    ]
    assert not development, f"checkout-only files entered the package: {development}"
    data = {name for name in files if not name.endswith(".py")}
    assert data == PACKAGE_DATA_FILES, (
        f"package data inventory differs: missing={sorted(PACKAGE_DATA_FILES - data)}, "
        f"unexpected={sorted(data - PACKAGE_DATA_FILES)}"
    )


def check_retired_forwarders(members) -> None:
    unexpected = RETIRED_FORWARDERS.intersection(members)
    assert not unexpected, f"retired forwarding modules in distribution: {sorted(unexpected)}"


# Exact distribution contract for the reviewed PDE foundation.
PDE_FILES = {
    "stochvolmodels/pde_solvers/__init__.py",
    "stochvolmodels/pde_solvers/_prepared_forward.py",
    "stochvolmodels/pde_solvers/adi.py",
    "stochvolmodels/pde_solvers/boundaries.py",
    "stochvolmodels/pde_solvers/coefficients.py",
    "stochvolmodels/pde_solvers/compound_poisson.py",
    "stochvolmodels/pde_solvers/diagnostics.py",
    "stochvolmodels/pde_solvers/exponential_jumps.py",
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
    "stochvolmodels/fitters/sparse_local_vol.py",
    "stochvolmodels/models/basket/__init__.py",
    "stochvolmodels/models/basket/model.py",
    "stochvolmodels/models/basket/simulation.py",
    "stochvolmodels/products/accumulator.py",
    "stochvolmodels/products/barrier_reverse_convertible.py",
    "stochvolmodels/pricers/accumulator.py",
    "stochvolmodels/pricers/barrier_reverse_convertible.py",
    "stochvolmodels/pricers/_barrier_monitoring.py",
    "stochvolmodels/fitters/logsv_backbone.py",
    "stochvolmodels/models/hawkes_jd/params.py",
    "stochvolmodels/models/hawkes_jd/transforms.py",
    "stochvolmodels/models/heston/model.py",
    "stochvolmodels/models/heston/params.py",
    "stochvolmodels/models/logsv/__init__.py",
    "stochvolmodels/models/logsv/coefficients.py",
    "stochvolmodels/models/logsv/grids.py",
    "stochvolmodels/models/logsv/kernel.py",
    "stochvolmodels/models/lsv/__init__.py",
    "stochvolmodels/models/lsv/simulation.py",
    "stochvolmodels/models/regime_logsv/equilibrium.py",
    "stochvolmodels/models/regime_logsv/params.py",
    "stochvolmodels/models/regime_logsv/simulation.py",
    "stochvolmodels/models/tdist/__init__.py",
    "stochvolmodels/models/tgarch/model.py",
    "stochvolmodels/models/tgarch/params.py",
    "stochvolmodels/models/tgarch/simulation.py",
    "stochvolmodels/pricers/tdist/__init__.py",
    "stochvolmodels/pricers/tdist/analytics.py",
    "stochvolmodels/data/density.py",
    "stochvolmodels/fitters/lsv_projection.py",
    "stochvolmodels/models/base.py",
    "stochvolmodels/models/gmm/__init__.py",
    "stochvolmodels/models/gmm/gmm.py",
    "stochvolmodels/models/hawkes_jd/__init__.py",
    "stochvolmodels/models/heston/__init__.py",
    "stochvolmodels/models/heston/kernels.py",
    "stochvolmodels/models/logsv/affine_expansion.py",
    "stochvolmodels/models/logsv/params.py",
    "stochvolmodels/models/logsv/simulation.py",
    "stochvolmodels/models/logsv/moments.py",
    "stochvolmodels/models/lsv/propagation.py",
    "stochvolmodels/models/rough_logsv/__init__.py",
    "stochvolmodels/models/rough_logsv/_classical_pde.py",
    "stochvolmodels/models/rough_logsv/_constrained_kernel.py",
    "stochvolmodels/models/rough_logsv/_history.py",
    "stochvolmodels/models/rough_logsv/_history_fast.py",
    "stochvolmodels/models/rough_logsv/expm.py",
    "stochvolmodels/models/rough_logsv/history.py",
    "stochvolmodels/models/rough_logsv/kernel_fit.py",
    "stochvolmodels/models/rough_logsv/multilevel.py",
    "stochvolmodels/models/rough_logsv/params.py",
    "stochvolmodels/models/rough_logsv/rough_kernel.py",
    "stochvolmodels/models/rough_logsv/split_simulation.py",
    "stochvolmodels/models/tdist/distribution.py",
    "stochvolmodels/models/tdist/params.py",
    "stochvolmodels/pricers/rough_logsv_pricer.py",
    "stochvolmodels/pricers/rough_logsv/__init__.py",
    "stochvolmodels/pricers/rough_logsv/split_simulation.py",
    "stochvolmodels/__init__.py",
    "stochvolmodels/data/model_paths.py",
    "stochvolmodels/models/__init__.py",
    "stochvolmodels/models/inverse_gamma_normal.py",
    "stochvolmodels/models/inverse_gamma/__init__.py",
    "stochvolmodels/models/inverse_gamma/terminal.py",
    "stochvolmodels/pricers/inverse_gamma_pricer.py",
    "stochvolmodels/models/logsv/model.py",
    "stochvolmodels/fitters/dupire.py",
    "stochvolmodels/fitters/lsv_fixed_point.py",
    "stochvolmodels/fitters/lsv_parametric.py",
    "stochvolmodels/fitters/lsv_market.py",
    "stochvolmodels/fitters/lsv_market_pipeline.py",
    "stochvolmodels/models/lsv/local_vol.py",
    "stochvolmodels/models/lsv/model.py",
    "stochvolmodels/pricers/density_pricer.py",
    "stochvolmodels/pricers/lsv_mc_pricer.py",
    "stochvolmodels/pricers/local_vol_pde_pricer.py",
    "stochvolmodels/pricers/logsv_pde_pricer.py",
    "stochvolmodels/models/regime_logsv/__init__.py",
    "stochvolmodels/models/tgarch/__init__.py",
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
    check_package_payload(members)
    check_retired_forwarders(members)
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
    extra_baselines = {name for name in members if name.endswith(".npz")} - PACKAGE_DATA_FILES
    assert not extra_baselines, f"unexpected regression baselines: {sorted(extra_baselines)}"

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
