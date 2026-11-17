"""
Production stress matrix and realistic benchmark for the historical LSV pipeline.

Scenario records vary grid anisotropy, correlation, vol-of-vol, leverage wings, time steps,
Monte Carlo paths, and jump settings. Benchmarks distinguish expected success from expected
detection and retain workload, timing, calibration, PDE, and MC evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.pde_solvers.adi import ADIConfig, ADIScheme
from stochvolmodels.pde_solvers.boundaries import BoundaryPair1D, BoundarySet2D
from stochvolmodels.pde_solvers.coefficients import PDE2DCoefficients
from stochvolmodels.pricers.density_pricer import (
    CoordinateSystem,
    DensityPricer,
    DensityState2D,
)
from stochvolmodels.pde_solvers.grids import Grid1D, TensorGrid2D
from stochvolmodels.pde_solvers.jumps import (
    TranslationJump,
    log_forward_jump_compensator,
)
from validation.pde_reference.lsv import LeverageSurface, LSVCoefficientProvider
from validation.pde_reference.lsv_mc import (
    LSVMonteCarloConfig,
    LSVMonteCarloPricer,
    LSVMonteCarloPricingResult,
)
from validation.pde_reference.market_pipeline import (
    LSVMarketPipeline,
    LSVMarketPipelineConfig,
    LSVMarketPipelineResult,
    btc_20211021_reference_config,
    load_frozen_btc_20211021_snapshot,
)
from stochvolmodels.pde_solvers.operators_1d import PDEDirection
from stochvolmodels.pde_solvers.solver_1d import DensityPolicy
from stochvolmodels.pde_solvers.solver_2d import PDE2DADISolver, PDEDiagnostics2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams

FloatArray = NDArray[np.float64]


class LSVStressExpectation(str, Enum):
    """Expected production outcome of one deliberately configured stress case."""

    PASS = "pass"
    DETECT = "detect"


@dataclass(frozen=True, slots=True)
class LSVPropagationStressScenario:
    """Fully explicit PDE/MC stress scenario and its numerical gates."""

    name: str
    params: LogSvParams
    grid: TensorGrid2D
    leverage_surface: LeverageSurface
    maturities: FloatArray
    strikes: FloatArray
    pde_steps_per_interval: tuple[int, ...]
    monte_carlo_config: LSVMonteCarloConfig
    jump: TranslationJump | None = None
    price_discretization_tolerance: float = 1.5e-3
    max_pde_forward_error: float = 3.0e-3
    max_pde_mass_error: float = 1.0e-8
    max_pde_negative_mass: float = 2.0e-4
    max_abs_mc_forward_z: float = 4.0
    expectation: LSVStressExpectation = LSVStressExpectation.PASS
    required_failed_checks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVPropagationStressScenario`` instance."""
        name = str(self.name).strip()
        if not name:
            raise ValueError("stress scenario name must be non-empty")
        if not isinstance(self.grid, TensorGrid2D):
            raise TypeError("grid must be a TensorGrid2D")
        if not isinstance(self.leverage_surface, LeverageSurface):
            raise TypeError("leverage_surface must be a LeverageSurface")
        if not np.array_equal(self.grid.x.nodes, self.leverage_surface.grid.nodes):
            raise ValueError("stress PDE and leverage x grids must match")
        maturities = _readonly_vector(self.maturities, "maturities")
        strikes = _readonly_vector(self.strikes, "strikes")
        if np.any(maturities <= 0.0) or np.any(np.diff(maturities) <= 0.0):
            raise ValueError("maturities must be positive and strictly increasing")
        if not np.array_equal(maturities, self.leverage_surface.times):
            raise ValueError("stress maturities must match leverage buckets")
        if np.any(strikes <= 0.0) or np.any(np.diff(strikes) <= 0.0):
            raise ValueError("strikes must be positive and strictly increasing")
        steps = _positive_integer_tuple(
            self.pde_steps_per_interval, "pde_steps_per_interval"
        )
        if len(steps) != maturities.size:
            raise ValueError("pde_steps_per_interval must match maturities")
        if not isinstance(self.monte_carlo_config, LSVMonteCarloConfig):
            raise TypeError("monte_carlo_config must be an LSVMonteCarloConfig")
        if self.jump is not None:
            if not isinstance(self.jump, TranslationJump):
                raise TypeError("jump must be a TranslationJump or None")
            if not np.array_equal(self.jump.grid.nodes, self.grid.x.nodes):
                raise ValueError("jump and stress PDE x grids must match")
        for field_name in (
            "price_discretization_tolerance",
            "max_pde_forward_error",
            "max_pde_mass_error",
            "max_pde_negative_mass",
            "max_abs_mc_forward_z",
        ):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
            object.__setattr__(self, field_name, value)
        if not isinstance(self.expectation, LSVStressExpectation):
            raise TypeError("expectation must be an LSVStressExpectation")
        required = tuple(str(value) for value in self.required_failed_checks)
        if self.expectation is LSVStressExpectation.PASS and required:
            raise ValueError("passing stress scenarios cannot require failed checks")
        if self.expectation is LSVStressExpectation.DETECT and not required:
            raise ValueError("detection scenarios must identify required failed checks")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "maturities", maturities)
        object.__setattr__(self, "strikes", strikes)
        object.__setattr__(self, "pde_steps_per_interval", steps)
        object.__setattr__(self, "required_failed_checks", required)

    @property
    def expected_shock_correlation(self) -> float:
        """Model-implied correlation between spot and total volatility shocks."""
        total = float(np.hypot(self.params.beta, self.params.volvol))
        return float(self.params.beta / total) if total > 0.0 else np.nan


@dataclass(frozen=True, slots=True)
class LSVPropagationStressResult:
    """PDE/MC prices, density diagnostics, and expectation status for one stress."""

    scenario: LSVPropagationStressScenario
    pde_calls: FloatArray
    pde_forward_errors: FloatArray
    pde_mass_errors: FloatArray
    pde_negative_masses: FloatArray
    pde_diagnostics: tuple[tuple[PDEDiagnostics2D, ...], ...]
    monte_carlo_result: LSVMonteCarloPricingResult
    elapsed_seconds: float
    checks: tuple[tuple[str, bool], ...] = field(init=False)

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVPropagationStressResult`` instance."""
        if not isinstance(self.scenario, LSVPropagationStressScenario):
            raise TypeError("scenario must be an LSVPropagationStressScenario")
        if not isinstance(self.monte_carlo_result, LSVMonteCarloPricingResult):
            raise TypeError("monte_carlo_result must be an LSVMonteCarloPricingResult")
        calls = _readonly_matrix(self.pde_calls, "pde_calls")
        forward_errors = _readonly_vector(self.pde_forward_errors, "pde_forward_errors")
        mass_errors = _readonly_vector(self.pde_mass_errors, "pde_mass_errors")
        negative_masses = _readonly_vector(
            self.pde_negative_masses, "pde_negative_masses"
        )
        expected_shape = (self.scenario.maturities.size, self.scenario.strikes.size)
        if calls.shape != expected_shape:
            raise ValueError("pde_calls must have maturity-by-strike shape")
        if not (
            forward_errors.shape
            == mass_errors.shape
            == negative_masses.shape
            == (expected_shape[0],)
        ):
            raise ValueError("PDE maturity diagnostics must align")
        diagnostics = tuple(tuple(values) for values in self.pde_diagnostics)
        if len(diagnostics) != expected_shape[0] or any(not values for values in diagnostics):
            raise ValueError("one non-empty PDE diagnostic tuple is required per maturity")
        elapsed = float(self.elapsed_seconds)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")

        mc_calls = np.vstack([item.calls for item in self.monte_carlo_result.slices])
        mc_errors = np.vstack(
            [item.call_standard_errors for item in self.monte_carlo_result.slices]
        )
        price_allowed = (
            4.0 * mc_errors + self.scenario.price_discretization_tolerance
        )
        forward_z = np.asarray(
            [item.forward_martingale_z_score for item in self.monte_carlo_result.slices]
        )
        checks = [
            ("pde_mc_prices", bool(np.all(np.abs(calls - mc_calls) <= price_allowed))),
            (
                "pde_forward",
                bool(np.max(forward_errors) <= self.scenario.max_pde_forward_error),
            ),
            ("pde_mass", bool(np.max(mass_errors) <= self.scenario.max_pde_mass_error)),
            (
                "pde_negative_mass",
                bool(np.max(negative_masses) <= self.scenario.max_pde_negative_mass),
            ),
            (
                "mc_martingale",
                bool(np.max(np.abs(forward_z)) <= self.scenario.max_abs_mc_forward_z),
            ),
        ]
        if self.scenario.jump is not None:
            path_count = self.monte_carlo_result.paths.path_count
            expected_counts = self.scenario.jump.intensity * self.scenario.maturities
            observed_counts = np.asarray(
                [item.mean_jump_count for item in self.monte_carlo_result.slices]
            )
            count_error = np.sqrt(expected_counts / path_count)
            checks.append(
                (
                    "jump_count",
                    bool(np.all(np.abs(observed_counts - expected_counts) <= 5.0 * count_error)),
                )
            )
        object.__setattr__(self, "pde_calls", calls)
        object.__setattr__(self, "pde_forward_errors", forward_errors)
        object.__setattr__(self, "pde_mass_errors", mass_errors)
        object.__setattr__(self, "pde_negative_masses", negative_masses)
        object.__setattr__(self, "pde_diagnostics", diagnostics)
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "checks", tuple(checks))

    @property
    def passed(self) -> bool:
        """Whether every numerical gate passed."""
        return all(passed for _, passed in self.checks)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        """Names of failed numerical gates."""
        return tuple(name for name, passed in self.checks if not passed)

    @property
    def expectation_satisfied(self) -> bool:
        """Whether pass scenarios passed and adversarial cases failed required gates."""
        if self.scenario.expectation is LSVStressExpectation.PASS:
            return self.passed
        failed = set(self.failed_checks)
        return all(name in failed for name in self.scenario.required_failed_checks)

    @property
    def max_price_error(self) -> float:
        """Maximum absolute PDE/MC call-price difference."""
        mc_calls = np.vstack([item.calls for item in self.monte_carlo_result.slices])
        return float(np.max(np.abs(self.pde_calls - mc_calls)))


@dataclass(frozen=True, slots=True)
class LSVCalibrationBenchmarkResult:
    """Timed realistic calibration plus workload and convergence diagnostics."""

    pipeline_result: LSVMarketPipelineResult
    elapsed_seconds: float
    initial_parameters: FloatArray
    optimizer_evaluations_by_maturity: NDArray[np.int64]
    pde_replays_by_maturity: NDArray[np.int64]
    expected_acceptance: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVCalibrationBenchmarkResult`` instance."""
        if not isinstance(self.pipeline_result, LSVMarketPipelineResult):
            raise TypeError("pipeline_result must be an LSVMarketPipelineResult")
        elapsed = float(self.elapsed_seconds)
        if not np.isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError("elapsed_seconds must be finite and positive")
        initial = _readonly_matrix(self.initial_parameters, "initial_parameters")
        evaluations = np.array(
            self.optimizer_evaluations_by_maturity,
            dtype=np.int64,
            copy=True,
            order="C",
        )
        replays = np.array(
            self.pde_replays_by_maturity,
            dtype=np.int64,
            copy=True,
            order="C",
        )
        maturity_count = self.pipeline_result.snapshot.ttms.size
        if (
            initial.shape[0] != maturity_count
            or evaluations.shape != (maturity_count,)
            or replays.shape != (maturity_count,)
        ):
            raise ValueError("benchmark diagnostics must contain one row/value per maturity")
        if np.any(evaluations < 1) or np.any(replays < evaluations):
            raise ValueError("optimizer counts must be positive and not exceed PDE replays")
        if not isinstance(self.expected_acceptance, bool):
            raise TypeError("expected_acceptance must be bool")
        evaluations.setflags(write=False)
        replays.setflags(write=False)
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "initial_parameters", initial)
        object.__setattr__(self, "optimizer_evaluations_by_maturity", evaluations)
        object.__setattr__(self, "pde_replays_by_maturity", replays)

    @property
    def total_optimizer_evaluations(self) -> int:
        """Total optimizer objective evaluations reported across maturities."""
        return int(np.sum(self.optimizer_evaluations_by_maturity))

    @property
    def total_pde_replays(self) -> int:
        """Candidate objective evolutions plus one independent final replay per slice."""
        return int(np.sum(self.pde_replays_by_maturity))

    @property
    def expectation_satisfied(self) -> bool:
        """Whether the benchmark acceptance status matches its declared expectation."""
        return self.pipeline_result.acceptance.passed is self.expected_acceptance

    def audit_summary(self) -> dict[str, str | float | bool]:
        """Return workload, timing, and acceptance metrics for benchmark logs."""
        result = self.pipeline_result
        summary = result.audit_summary()
        summary.update(
            {
                "elapsed_seconds": self.elapsed_seconds,
                "optimizer_evaluations": float(self.total_optimizer_evaluations),
                "pde_replays": float(self.total_pde_replays),
                "calibration_grid_nodes": float(
                    np.prod(result.calibration_result.initial_density.grid.shape)
                ),
                "repricing_grid_nodes": float(np.prod(result.pde_repricing.grid.shape)),
                "monte_carlo_paths_per_simulation": float(
                    result.monte_carlo_validation.normalized_pricing.monte_carlo_result.paths.path_count
                ),
                "monte_carlo_simulations": 2.0,
                "expected_acceptance": self.expected_acceptance,
                "expectation_satisfied": self.expectation_satisfied,
            }
        )
        return summary


@dataclass(frozen=True, slots=True)
class _JumpCompensatedLSVProvider:
    """Add the fixed-jump martingale compensator to continuous LSV x drift."""

    base: LSVCoefficientProvider
    compensator: float

    @property
    def is_time_dependent(self) -> bool:
        """Return whether the coefficient provider varies with time."""
        return self.base.is_time_dependent

    def evaluate(self, time: float, grid: TensorGrid2D) -> PDE2DCoefficients:
        """Evaluate coefficients on the requested tensor grid and time."""
        coefficients = self.base.evaluate(time, grid)
        return PDE2DCoefficients(
            variance_x=coefficients.variance_x,
            drift_x=coefficients.drift_x + self.compensator,
            variance_y=coefficients.variance_y,
            drift_y=coefficients.drift_y,
            covariance_xy=coefficients.covariance_xy,
            discount=coefficients.discount,
        )


def run_lsv_propagation_stress(
    scenario: LSVPropagationStressScenario,
) -> LSVPropagationStressResult:
    """Run one stress using independent 2-D PDE propagation and LSV Monte Carlo."""
    if not isinstance(scenario, LSVPropagationStressScenario):
        raise TypeError("scenario must be an LSVPropagationStressScenario")
    started = perf_counter()
    grid = scenario.grid
    start_density = DensityState2D(
        grid,
        grid.project_delta(0.0, scenario.params.sigma0),
        0.0,
        CoordinateSystem.LOG_FORWARD,
    )
    solver = PDE2DADISolver(
        ADIConfig(theta=1.0, scheme=ADIScheme.DOUGLAS),
        DensityPolicy(
            negativity_tolerance=scenario.max_pde_negative_mass,
            mass_error_tolerance=scenario.max_pde_mass_error,
            renormalize=False,
            fail_on_violation=False,
        ),
    )
    base_provider = LSVCoefficientProvider(scenario.params, scenario.leverage_surface)
    if scenario.jump is None:
        provider = base_provider
    else:
        provider = _JumpCompensatedLSVProvider(
            base_provider,
            log_forward_jump_compensator(
                scenario.jump.intensity,
                scenario.jump.jump_size,
            ),
        )
    boundaries = BoundarySet2D(
        x=BoundaryPair1D.natural(),
        y=BoundaryPair1D.natural(),
    )
    pde_calls: list[FloatArray] = []
    forward_errors: list[float] = []
    mass_errors: list[float] = []
    negative_masses: list[float] = []
    diagnostics_by_maturity: list[tuple[PDEDiagnostics2D, ...]] = []
    for maturity, step_count in zip(
        scenario.maturities, scenario.pde_steps_per_interval
    ):
        times = np.linspace(start_density.time, maturity, step_count + 1)
        if scenario.jump is None:
            solution = solver.solve(
                grid,
                provider,
                boundaries,
                PDEDirection.FORWARD,
                times,
                start_density.values,
            )
            terminal_values = solution.values
            diagnostics = solution.diagnostics
        else:
            terminal_values, diagnostics = _solve_jump_interval(
                solver,
                grid,
                provider,
                boundaries,
                scenario.jump,
                times,
                start_density.values,
            )
        terminal = DensityState2D(
            grid,
            terminal_values,
            maturity,
            CoordinateSystem.LOG_FORWARD,
        )
        pricer = DensityPricer(terminal.marginal_x_state(), forward=1.0)
        pde_calls.append(pricer.price_slice(scenario.strikes).calls)
        forward_moment = grid.integrate(np.exp(grid.x_mesh) * terminal.values)
        forward_errors.append(abs(forward_moment - 1.0))
        mass_errors.append(abs(terminal.mass - 1.0))
        negative_masses.append(terminal.negative_mass)
        diagnostics_by_maturity.append(diagnostics)
        start_density = terminal

    monte_carlo = LSVMonteCarloPricer(
        scenario.params,
        scenario.leverage_surface,
        scenario.monte_carlo_config,
        jump=scenario.jump,
    ).price(scenario.strikes, maturities=scenario.maturities)
    return LSVPropagationStressResult(
        scenario=scenario,
        pde_calls=np.vstack(pde_calls),
        pde_forward_errors=np.asarray(forward_errors),
        pde_mass_errors=np.asarray(mass_errors),
        pde_negative_masses=np.asarray(negative_masses),
        pde_diagnostics=tuple(diagnostics_by_maturity),
        monte_carlo_result=monte_carlo,
        elapsed_seconds=perf_counter() - started,
    )


def run_production_lsv_stress_matrix() -> tuple[LSVPropagationStressResult, ...]:
    """Run correlation, vol-of-vol, wings, jumps, sparse-time, and grid stresses."""
    return tuple(run_lsv_propagation_stress(item) for item in production_lsv_stress_scenarios())


def production_lsv_stress_scenarios() -> tuple[LSVPropagationStressScenario, ...]:
    """Return the fully explicit production stress matrix without running it."""
    base_maturities = np.array([0.15, 0.40])
    strikes = np.array([0.82, 0.92, 1.00, 1.08, 1.22])
    scenarios: list[LSVPropagationStressScenario] = []

    negative_grid = _stress_grid(-1.9, 1.6, 101, 0.025, 2.2, 71, 0.25)
    negative_params = LogSvParams(
        sigma0=0.24,
        theta=0.25,
        kappa1=3.0,
        kappa2=8.0,
        beta=-0.58,
        volvol=0.10,
    )
    scenarios.append(
        _scenario(
            "extreme_negative_correlation",
            negative_params,
            negative_grid,
            base_maturities,
            strikes,
            (80, 120),
            _regular_leverage(negative_grid.x, base_maturities),
            expectation=LSVStressExpectation.DETECT,
            required_failed_checks=("pde_negative_mass",),
        )
    )

    positive_grid = _stress_grid(-1.8, 1.8, 101, 0.025, 2.2, 71, 0.25)
    positive_params = LogSvParams(
        sigma0=0.24,
        theta=0.25,
        kappa1=3.0,
        kappa2=8.0,
        beta=0.58,
        volvol=0.10,
    )
    scenarios.append(
        _scenario(
            "extreme_positive_correlation",
            positive_params,
            positive_grid,
            base_maturities,
            strikes,
            (80, 120),
            _regular_leverage(positive_grid.x, base_maturities),
            expectation=LSVStressExpectation.DETECT,
            required_failed_checks=("pde_negative_mass",),
        )
    )

    volvol_grid = _stress_grid(-2.1, 1.8, 111, 0.015, 2.8, 81, 0.24)
    volvol_params = LogSvParams(
        sigma0=0.23,
        theta=0.25,
        kappa1=3.5,
        kappa2=12.0,
        beta=-0.30,
        volvol=1.05,
    )
    scenarios.append(
        _scenario(
            "high_vol_of_vol",
            volvol_params,
            volvol_grid,
            base_maturities,
            strikes,
            (100, 160),
            _regular_leverage(volvol_grid.x, base_maturities),
            price_tolerance=2.5e-3,
            max_negative_mass=7.5e-4,
        )
    )

    wing_grid = _stress_grid(-2.0, 1.7, 111, 0.025, 1.8, 71, 0.24)
    wing_params = LogSvParams(
        sigma0=0.24,
        theta=0.25,
        kappa1=3.0,
        kappa2=8.0,
        beta=-0.22,
        volvol=0.42,
    )
    scenarios.append(
        _scenario(
            "steep_leverage_wings",
            wing_params,
            wing_grid,
            base_maturities,
            strikes,
            (90, 140),
            _wing_leverage(wing_grid.x, base_maturities),
            price_tolerance=2.5e-3,
        )
    )

    jump_grid = _stress_grid(-2.2, 1.8, 111, 0.025, 1.8, 71, 0.24)
    jump_params = LogSvParams(
        sigma0=0.24,
        theta=0.25,
        kappa1=3.0,
        kappa2=8.0,
        beta=-0.22,
        volvol=0.42,
    )
    jump_leverage = _regular_leverage(jump_grid.x, base_maturities)
    jump = TranslationJump(jump_grid.x, intensity=2.0, jump_size=-0.18)
    scenarios.append(
        _scenario(
            "compensated_down_jumps",
            jump_params,
            jump_grid,
            base_maturities,
            strikes,
            (90, 150),
            jump_leverage,
            jump=jump,
            price_tolerance=3.0e-3,
        )
    )

    sparse_maturities = np.array([0.03, 0.75])
    sparse_grid = _stress_grid(-2.4, 2.0, 121, 0.02, 2.0, 75, 0.24)
    sparse_params = LogSvParams(
        sigma0=0.24,
        theta=0.25,
        kappa1=2.5,
        kappa2=8.0,
        beta=-0.28,
        volvol=0.50,
    )
    scenarios.append(
        _scenario(
            "sparse_maturity_buckets",
            sparse_params,
            sparse_grid,
            sparse_maturities,
            strikes,
            (60, 260),
            _wing_leverage(sparse_grid.x, sparse_maturities),
            price_tolerance=3.0e-3,
        )
    )

    narrow_maturities = np.array([0.20, 0.60])
    narrow_grid = _stress_grid(-0.24, 0.24, 61, 0.12, 0.38, 41, 0.24)
    narrow_params = LogSvParams(
        sigma0=0.24,
        theta=0.25,
        kappa1=3.0,
        kappa2=8.0,
        beta=-0.28,
        volvol=0.50,
    )
    scenarios.append(
        _scenario(
            "narrow_grid_expected_detection",
            narrow_params,
            narrow_grid,
            narrow_maturities,
            strikes,
            (100, 180),
            _regular_leverage(narrow_grid.x, narrow_maturities),
            expectation=LSVStressExpectation.DETECT,
            required_failed_checks=("pde_mc_prices",),
        )
    )
    return tuple(scenarios)


def difficult_btc_initial_parameters() -> FloatArray:
    """Return bounded, deliberately poor product-family starts for the BTC benchmark."""
    values = np.array(
        [
            [0.35, -2.50, 8.0],
            [3.20, 4.50, 12.0],
            [0.40, -1.50, 15.0],
            [3.50, 5.50, 18.0],
        ]
    )
    values.setflags(write=False)
    return values


def run_realistic_btc_calibration_benchmark() -> LSVCalibrationBenchmarkResult:
    """Benchmark the accepted four-maturity stochastic LSV production pilot."""
    config = btc_20211021_reference_config()
    started = perf_counter()
    result = LSVMarketPipeline(config).run(load_frozen_btc_20211021_snapshot())
    elapsed = perf_counter() - started
    initial = np.vstack(
        [item.initial_parameters for item in result.calibration_result.slices]
    )
    evaluations = np.asarray(
        [item.optimizer_evaluations_count for item in result.calibration_result.slices],
        dtype=np.int64,
    )
    replays = np.asarray(
        [len(item.optimizer_evaluations) + 1 for item in result.calibration_result.slices],
        dtype=np.int64,
    )
    return LSVCalibrationBenchmarkResult(
        pipeline_result=result,
        elapsed_seconds=elapsed,
        initial_parameters=initial,
        optimizer_evaluations_by_maturity=evaluations,
        pde_replays_by_maturity=replays,
        expected_acceptance=True,
    )


def run_difficult_btc_initialization_stress() -> LSVCalibrationBenchmarkResult:
    """Run the realistic calibration from adverse starts and require gate detection."""
    initial = difficult_btc_initial_parameters()
    reference = btc_20211021_reference_config()
    calibration_config = replace(
        reference.calibration_config,
        maximum_evaluations=24,
    )
    config: LSVMarketPipelineConfig = replace(
        reference,
        calibration_config=calibration_config,
        calibration_initial_parameters=initial,
    )
    started = perf_counter()
    result = LSVMarketPipeline(config).run(load_frozen_btc_20211021_snapshot())
    elapsed = perf_counter() - started
    evaluations = np.asarray(
        [item.optimizer_evaluations_count for item in result.calibration_result.slices],
        dtype=np.int64,
    )
    replays = np.asarray(
        [len(item.optimizer_evaluations) + 1 for item in result.calibration_result.slices],
        dtype=np.int64,
    )
    return LSVCalibrationBenchmarkResult(
        pipeline_result=result,
        elapsed_seconds=elapsed,
        initial_parameters=initial,
        optimizer_evaluations_by_maturity=evaluations,
        pde_replays_by_maturity=replays,
        expected_acceptance=False,
    )


def _solve_jump_interval(
    solver: PDE2DADISolver,
    grid: TensorGrid2D,
    provider: _JumpCompensatedLSVProvider,
    boundaries: BoundarySet2D,
    jump: TranslationJump,
    times: FloatArray,
    initial_values: FloatArray,
) -> tuple[FloatArray, tuple[PDEDiagnostics2D, ...]]:
    """Advance one stress state across a configured jump interval."""
    state = np.asarray(initial_values, dtype=np.float64)
    diagnostics: list[PDEDiagnostics2D] = []
    for time_start, time_end in zip(times[:-1], times[1:]):
        step = float(time_end - time_start)
        state = jump.poisson_step_tensor(
            state,
            grid,
            axis=0,
            direction=PDEDirection.FORWARD,
            step_size=0.5 * step,
        )
        solution = solver.solve(
            grid,
            provider,
            boundaries,
            PDEDirection.FORWARD,
            np.array([time_start, time_end]),
            state,
        )
        state = jump.poisson_step_tensor(
            solution.values,
            grid,
            axis=0,
            direction=PDEDirection.FORWARD,
            step_size=0.5 * step,
        )
        diagnostics.extend(solution.diagnostics)
    return np.asarray(state, dtype=np.float64), tuple(diagnostics)


def _stress_grid(
    x_lower: float,
    x_upper: float,
    x_size: int,
    y_lower: float,
    y_upper: float,
    y_size: int,
    sigma_center: float,
) -> TensorGrid2D:
    """Construct the tensor grid requested by a stress scenario."""
    return TensorGrid2D(
        Grid1D.concentrated(x_lower, x_upper, x_size, center=0.0, scale=0.10),
        Grid1D.concentrated(
            y_lower,
            y_upper,
            y_size,
            center=sigma_center,
            scale=0.04,
        ),
    )


def _regular_leverage(grid: Grid1D, maturities: FloatArray) -> LeverageSurface:
    """Construct the regular leverage surface used by stress scenarios."""
    rows = []
    for index, _ in enumerate(maturities):
        rows.append(
            1.0
            + 0.025 * index
            - (0.16 + 0.04 * index) * np.tanh(3.0 * grid.nodes)
            + (0.04 + 0.03 * index) * grid.nodes**2
        )
    return LeverageSurface(maturities, grid, np.vstack(rows))


def _wing_leverage(grid: Grid1D, maturities: FloatArray) -> LeverageSurface:
    """Construct the stressed leverage surface with amplified wings."""
    rows = []
    for index, _ in enumerate(maturities):
        raw = (
            1.0
            + 0.04 * index
            - (0.38 + 0.08 * index) * np.tanh(4.0 * grid.nodes)
            + (0.30 + 0.08 * index) * grid.nodes**2
        )
        rows.append(np.clip(raw, 0.35, 2.80))
    return LeverageSurface(
        maturities,
        grid,
        np.vstack(rows),
        leverage_floor=0.30,
        leverage_cap=2.90,
    )


def _scenario(
    name: str,
    params: LogSvParams,
    grid: TensorGrid2D,
    maturities: FloatArray,
    strikes: FloatArray,
    pde_steps: tuple[int, ...],
    leverage: LeverageSurface,
    *,
    jump: TranslationJump | None = None,
    price_tolerance: float = 1.5e-3,
    max_negative_mass: float = 2.0e-4,
    expectation: LSVStressExpectation = LSVStressExpectation.PASS,
    required_failed_checks: tuple[str, ...] = (),
) -> LSVPropagationStressScenario:
    """Construct one fully specified production stress scenario."""
    return LSVPropagationStressScenario(
        name=name,
        params=params,
        grid=grid,
        leverage_surface=leverage,
        maturities=maturities,
        strikes=strikes,
        pde_steps_per_interval=pde_steps,
        monte_carlo_config=LSVMonteCarloConfig(
            paths=80_000,
            steps_per_year=720,
            seed=20260818,
            antithetic=True,
            confidence_level=0.95,
        ),
        jump=jump,
        price_discretization_tolerance=price_tolerance,
        max_pde_forward_error=3.0e-3,
        max_pde_mass_error=1.0e-8,
        max_pde_negative_mass=max_negative_mass,
        max_abs_mc_forward_z=4.0,
        expectation=expectation,
        required_failed_checks=required_failed_checks,
    )


def _positive_integer_tuple(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    """Validate and return strictly positive integer tuple."""
    result = tuple(values)
    if not result or any(
        isinstance(value, bool) or int(value) != value or int(value) < 1 for value in result
    ):
        raise ValueError(f"{name} must contain positive integers")
    return tuple(int(value) for value in result)


def _readonly_vector(values: ArrayLike, name: str) -> FloatArray:
    """Validate vector and return an immutable NumPy value."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    array.setflags(write=False)
    return array


def _readonly_matrix(values: ArrayLike, name: str) -> FloatArray:
    """Validate matrix and return an immutable NumPy value."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite two-dimensional array")
    array.setflags(write=False)
    return array
