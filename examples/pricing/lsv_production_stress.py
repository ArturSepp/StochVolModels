"""Replay seven fixed-parameter LSV stresses on the package PDE and Monte Carlo pricers.

Run from the checkout with `python -m examples.pricing.lsv_production_stress`.
Prices are undiscounted European calls per unit forward. The archived density gates
apply at quoted expiries. The peak over all completed PDE steps is reported separately.
"""

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.lsv import FixedLogForwardJump, LSVModel, LeverageSurface
from stochvolmodels.models.lsv_simulation import LSVMonteCarloConfig
from stochvolmodels.pde_solvers import ADIConfig, ADIScheme, Grid1D, TensorGrid2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer, LogSvPdeResult
from stochvolmodels.pricers.lsv_mc_pricer import LSVMonteCarloPricer


STRIKES = (0.82, 0.92, 1.00, 1.08, 1.22)
MC_CONFIG = LSVMonteCarloConfig(paths=80_000, steps_per_year=720, seed=20260818, antithetic=True)


@dataclass(frozen=True)
class StressScenario:
    """A fully specified grid, model, expiry partition and archived acceptance limits."""

    name: str
    grid: TensorGrid2D
    model: LSVModel
    maturities: tuple[float, ...]
    steps: tuple[int, ...]
    price_tolerance: float
    negative_mass_tolerance: float
    expected_failures: tuple[str, ...]


def _scenario(
    name,
    params,
    x_grid,
    sigma_grid,
    steps,
    *,
    maturities=(0.15, 0.40),
    wings=False,
    jump=None,
    price_tolerance=0.0015,
    negative_mass_tolerance=0.0002,
    expected_failures=(),
) -> StressScenario:
    """Construct the archived sinh grids and the two expiry buckets of spot leverage."""
    x = Grid1D.concentrated(*x_grid, center=0.0, scale=0.10)
    lower, upper, size, center = sigma_grid
    sigma = Grid1D.concentrated(lower, upper, size, center=center, scale=0.04)
    i = np.arange(len(maturities))[:, None]
    if wings:
        raw = (
            1 + 0.04 * i - (0.38 + 0.08 * i) * np.tanh(4 * x.nodes) + (0.30 + 0.08 * i) * x.nodes**2
        )
        surface = LeverageSurface(maturities, x, np.clip(raw, 0.35, 2.80), 0.30, 2.90)
    else:
        raw = (
            1
            + 0.025 * i
            - (0.16 + 0.04 * i) * np.tanh(3 * x.nodes)
            + (0.04 + 0.03 * i) * x.nodes**2
        )
        surface = LeverageSurface(maturities, x, raw)
    return StressScenario(
        name,
        TensorGrid2D(x, sigma),
        LSVModel.from_params(params, surface, jump=jump),
        maturities,
        steps,
        price_tolerance,
        negative_mass_tolerance,
        expected_failures,
    )


def production_lsv_stress_scenarios() -> tuple[StressScenario, ...]:
    """Return the seven frozen propagation scenarios in archive order."""
    base = LogSvParams(sigma0=0.24, theta=0.25, kappa1=3.0, kappa2=8.0, beta=-0.22, volvol=0.42)
    corr_sigma = (0.025, 2.2, 71, 0.25)
    wing_sigma = (0.025, 1.8, 71, 0.24)
    corr_failure = ("pde_negative_mass",)
    return (
        _scenario(
            "extreme_negative_correlation",
            replace(base, beta=-0.58, volvol=0.10),
            (-1.9, 1.6, 101),
            corr_sigma,
            (80, 120),
            expected_failures=corr_failure,
        ),
        _scenario(
            "extreme_positive_correlation",
            replace(base, beta=0.58, volvol=0.10),
            (-1.8, 1.8, 101),
            corr_sigma,
            (80, 120),
            expected_failures=corr_failure,
        ),
        _scenario(
            "high_vol_of_vol",
            replace(base, sigma0=0.23, kappa1=3.5, kappa2=12, beta=-0.30, volvol=1.05),
            (-2.1, 1.8, 111),
            (0.015, 2.8, 81, 0.24),
            (100, 160),
            price_tolerance=0.0025,
            negative_mass_tolerance=0.00075,
        ),
        _scenario(
            "steep_leverage_wings",
            base,
            (-2.0, 1.7, 111),
            wing_sigma,
            (90, 140),
            wings=True,
            price_tolerance=0.0025,
        ),
        _scenario(
            "compensated_down_jumps",
            base,
            (-2.2, 1.8, 111),
            wing_sigma,
            (90, 150),
            jump=FixedLogForwardJump(intensity=2.0, jump_size=-0.18),
            price_tolerance=0.003,
        ),
        _scenario(
            "sparse_maturity_buckets",
            replace(base, kappa1=2.5, beta=-0.28, volvol=0.50),
            (-2.4, 2.0, 121),
            (0.02, 2.0, 75, 0.24),
            (60, 260),
            maturities=(0.03, 0.75),
            wings=True,
            price_tolerance=0.003,
        ),
        _scenario(
            "narrow_grid_expected_detection",
            replace(base, beta=-0.28, volvol=0.50),
            (-0.24, 0.24, 61),
            (0.12, 0.38, 41, 0.24),
            (100, 180),
            maturities=(0.20, 0.60),
            expected_failures=("pde_mc_prices",),
        ),
    )


@dataclass(frozen=True)
class StressResult:
    """Raw prices, pair-aware Monte Carlo errors and completed-step PDE diagnostics."""

    scenario: StressScenario
    pde: LogSvPdeResult
    mc_prices: np.ndarray
    mc_standard_errors: np.ndarray
    mc_forward_z: np.ndarray
    mean_jump_counts: np.ndarray
    shock_correlation: float

    @property
    def price_errors(self) -> np.ndarray:
        """Absolute PDE/MC differences in expiry-by-strike order."""
        return np.abs(np.asarray(self.pde.prices) - self.mc_prices)

    @property
    def checks(self) -> dict[str, bool]:
        """Evaluate the archived inclusive gates on raw prices and expiry densities."""
        expiries = self.pde.expiries
        price_limit = 4 * self.mc_standard_errors + self.scenario.price_tolerance
        checks = {
            "pde_mc_prices": bool(np.all(self.price_errors <= price_limit)),
            "pde_forward": all(abs(e.normalized_forward - 1) <= 0.003 for e in expiries),
            "pde_mass": all(abs(e.mass - 1) <= 1e-8 for e in expiries),
            "pde_negative_mass": all(
                e.negative_mass <= self.scenario.negative_mass_tolerance for e in expiries
            ),
            "mc_martingale": bool(np.all(np.abs(self.mc_forward_z) <= 4)),
        }
        jump = self.scenario.model.jump
        if jump is not None:
            expected = jump.intensity * np.array(self.scenario.maturities)
            count_limit = 5 * np.sqrt(expected / MC_CONFIG.paths)
            checks["jump_count"] = bool(
                np.all(np.abs(self.mean_jump_counts - expected) <= count_limit)
            )
        return checks

    @property
    def failed_checks(self) -> tuple[str, ...]:
        """List failed numerical gates independently of the scenario expectation."""
        return tuple(name for name, passed in self.checks.items() if not passed)

    @property
    def matches_archive(self) -> bool:
        """Require exactly the archived failures, including for the three adverse cases."""
        return self.failed_checks == self.scenario.expected_failures


def run_lsv_propagation_stress(scenario: StressScenario) -> StressResult:
    """Price all expiries in one PDE solve and one independently seeded path simulation."""
    times = np.array(scenario.maturities)
    chain = OptionChain(
        ttms=times,
        ids=times.astype(str),
        forwards=np.ones(times.size),
        strikes_ttms=(np.array(STRIKES),) * times.size,
        optiontypes_ttms=(np.full(len(STRIKES), "C"),) * times.size,
    )
    pde = LogSvPdePricer(
        grid=scenario.grid,
        steps_per_interval=scenario.steps,
        quadratic_start=False,  # Archived stress decisions use uniform steps.
        adi=ADIConfig(theta=1.0, scheme=ADIScheme.DOUGLAS),
    ).price_chain_detailed(chain, scenario.model)
    mc = LSVMonteCarloPricer(config=MC_CONFIG).price_chain_detailed(chain, scenario.model)
    # All seven stochastic scenarios have positive forward sampling error.
    z = [(e.forward_mean - 1) / e.forward_standard_error for e in mc.expiries]
    return StressResult(
        scenario,
        pde,
        np.vstack(mc.prices),
        np.vstack(mc.standard_errors),
        np.array(z),
        mc.paths.states["jump_count"][:, 1:].mean(axis=0),
        mc.paths.diagnostics["diffusion_shock_correlation"],
    )


class Locals(Enum):
    """Offline stress runs."""

    PRODUCTION_MATRIX = 1


def run_local(local: Locals) -> None:
    """Print the seven decisions and fail if any differs from its archived outcome."""
    if local != Locals.PRODUCTION_MATRIX:
        raise ValueError(f"unknown stress case {local}")
    print(
        f"{'Scenario':32} {'|PDE-MC|':>10} {'|F-1|':>10} {'|mass-1|':>10} "
        f"{'Expiry neg.':>12} {'Peak neg.':>12}  Decision"
    )
    mismatches = []
    for scenario in production_lsv_stress_scenarios():
        result = run_lsv_propagation_stress(scenario)
        expiries = result.pde.expiries
        decision = (
            "reject (" + ", ".join(result.failed_checks) + ")" if result.failed_checks else "accept"
        )
        print(
            f"{scenario.name:32} {result.price_errors.max():10.4g} "
            f"{max(abs(e.normalized_forward - 1) for e in expiries):10.4g} "
            f"{max(abs(e.mass - 1) for e in expiries):10.4g} "
            f"{max(e.negative_mass for e in expiries):12.4g} "
            f"{result.pde.peak_negative_mass:12.4g}  {decision}"
        )
        if not result.matches_archive:
            mismatches.append(f"{scenario.name} (expected failures {scenario.expected_failures})")
    if mismatches:
        raise RuntimeError("stress decisions differ from archive: " + ", ".join(mismatches))


if __name__ == "__main__":
    run_local(local=Locals.PRODUCTION_MATRIX)
