"""
Frozen-market orchestration for the historical lognormal-LSV pipeline.

A frozen snapshot is normalized, passed through Dupire recovery and LSV calibration, repriced on
PDE grids, and compared with independent Monte Carlo. Fingerprints, repair fractions, spread
coverage, mass errors, IV errors, and acceptance thresholds form a single auditable result.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pde_solvers.adi import ADIConfig, ADIScheme
from stochvolmodels.pde_solvers.boundaries import BoundaryPair1D, BoundarySet2D
from validation.pde_reference.calibration import (
    CalibrationWeighting,
    LSVParametricCalibrationConfig,
    LSVParametricCalibrationResult,
    LSVParametricCalibrator,
)
from stochvolmodels.pricers.density_pricer import (
    CoordinateSystem,
    DensityPricer,
    DensityState2D,
)
from validation.pde_reference.dupire import (
    CallSurface,
    DupireRecoverer,
    DupireRepairPolicy,
    DupireResult,
)
from stochvolmodels.pde_solvers.grids import Grid1D, TensorGrid2D
from validation.pde_reference.integration import (
    LSVCalibrationInput,
    LSVOptionChainAdapter,
    LSVOptionChainPricer,
    LSVOptionChainPricingResult,
    LSVQuoteSelection,
)
from validation.pde_reference.local_vol import (
    LocalVolFamily,
    LocalVolParameterCodec,
    LocalVolSurface,
)
from stochvolmodels.models.logsv.coefficients import LogSvCoefficientProvider
from validation.pde_reference.lsv import LSVCoefficientProvider
from validation.pde_reference.lsv_mc import (
    LSVMonteCarloConfig,
    LSVMonteCarloPricer,
)
from stochvolmodels.pde_solvers.operators_1d import PDEDirection
from stochvolmodels.pde_solvers.solver_1d import DensityPolicy
from stochvolmodels.pde_solvers.solver_2d import PDE2DADISolver, PDEDiagnostics2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams

FloatArray = NDArray[np.float64]
StringArray = NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class FrozenOptionChainSnapshot:
    """Immutable bid/ask IV snapshot with a reproducible SHA-256 fingerprint."""

    snapshot_id: str
    as_of: str
    source: str
    ticker: str
    slice_ids: tuple[str, ...]
    ttms: FloatArray
    forwards: FloatArray
    discount_factors: FloatArray
    strikes: tuple[FloatArray, ...]
    option_types: tuple[StringArray, ...]
    bid_implied_volatilities: tuple[FloatArray, ...]
    ask_implied_volatilities: tuple[FloatArray, ...]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``FrozenOptionChainSnapshot`` instance."""
        text_fields = ("snapshot_id", "as_of", "source")
        for name in text_fields:
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "ticker", str(self.ticker).strip())
        slice_ids = tuple(str(value) for value in self.slice_ids)
        ttms = _readonly_vector(self.ttms, "ttms")
        forwards = _readonly_vector(self.forwards, "forwards")
        discounts = _readonly_vector(self.discount_factors, "discount_factors")
        strikes = _readonly_vector_tuple(self.strikes, "strikes")
        option_types = tuple(_readonly_strings(values) for values in self.option_types)
        bids = _readonly_vector_tuple(
            self.bid_implied_volatilities, "bid_implied_volatilities"
        )
        asks = _readonly_vector_tuple(
            self.ask_implied_volatilities, "ask_implied_volatilities"
        )
        maturity_count = ttms.size
        if not (
            len(slice_ids)
            == forwards.size
            == discounts.size
            == len(strikes)
            == len(option_types)
            == len(bids)
            == len(asks)
            == maturity_count
        ):
            raise ValueError("snapshot arrays must contain one entry per maturity")
        if np.any(np.diff(ttms) <= 0.0) or np.any(ttms <= 0.0):
            raise ValueError("ttms must be positive and strictly increasing")
        if np.any(forwards <= 0.0) or np.any(discounts <= 0.0):
            raise ValueError("forwards and discount factors must be positive")
        for strike_slice, type_slice, bid_slice, ask_slice in zip(
            strikes, option_types, bids, asks
        ):
            if not (
                strike_slice.shape
                == type_slice.shape
                == bid_slice.shape
                == ask_slice.shape
            ):
                raise ValueError("strike, option-type, and quote slices must align")
            if strike_slice.size < 2 or np.any(strike_slice <= 0.0):
                raise ValueError("every strike slice must contain positive values")
            if np.any(~np.isin(type_slice, ("C", "P"))):
                raise ValueError("frozen LSV snapshots accept only C/P option types")
            if np.any(bid_slice <= 0.0) or np.any(bid_slice > ask_slice):
                raise ValueError("bid/ask implied volatilities must be positive and ordered")
        object.__setattr__(self, "slice_ids", slice_ids)
        object.__setattr__(self, "ttms", ttms)
        object.__setattr__(self, "forwards", forwards)
        object.__setattr__(self, "discount_factors", discounts)
        object.__setattr__(self, "strikes", strikes)
        object.__setattr__(self, "option_types", option_types)
        object.__setattr__(self, "bid_implied_volatilities", bids)
        object.__setattr__(self, "ask_implied_volatilities", asks)
        object.__setattr__(self, "fingerprint", _snapshot_fingerprint(self))

    @classmethod
    def from_option_chain(
        cls,
        option_chain: OptionChain,
        *,
        snapshot_id: str,
        as_of: str,
        source: str,
    ) -> FrozenOptionChainSnapshot:
        """Copy an existing :class:`OptionChain` into immutable snapshot arrays."""
        if not isinstance(option_chain, OptionChain):
            raise TypeError("option_chain must be an OptionChain")
        if option_chain.bid_ivs is None or option_chain.ask_ivs is None:
            raise ValueError("frozen calibration snapshots require bid and ask IVs")
        slice_ids = (
            tuple(str(value) for value in option_chain.ids)
            if option_chain.ids is not None
            else tuple(f"T={value:.6g}" for value in option_chain.ttms)
        )
        return cls(
            snapshot_id=snapshot_id,
            as_of=as_of,
            source=source,
            ticker="" if option_chain.ticker is None else option_chain.ticker,
            slice_ids=slice_ids,
            ttms=option_chain.ttms,
            forwards=option_chain.forwards,
            discount_factors=option_chain.discfactors,
            strikes=tuple(option_chain.strikes_ttms),
            option_types=tuple(option_chain.optiontypes_ttms),
            bid_implied_volatilities=tuple(option_chain.bid_ivs),
            ask_implied_volatilities=tuple(option_chain.ask_ivs),
        )

    def to_option_chain(self) -> OptionChain:
        """Return a fresh mutable data-layer object without sharing snapshot arrays."""
        return OptionChain(
            ttms=self.ttms.copy(),
            forwards=self.forwards.copy(),
            strikes_ttms=tuple(values.copy() for values in self.strikes),
            optiontypes_ttms=tuple(values.copy() for values in self.option_types),
            ids=np.asarray(self.slice_ids, dtype=str),
            discfactors=self.discount_factors.copy(),
            ticker=self.ticker,
            bid_ivs=tuple(values.copy() for values in self.bid_implied_volatilities),
            ask_ivs=tuple(values.copy() for values in self.ask_implied_volatilities),
        )


@dataclass(frozen=True, slots=True)
class LSVMarketAcceptanceCriteria:
    """Numerical gates applied after calibration, fine-grid replay, and MC."""

    max_dupire_repaired_fraction: float = 0.15
    max_pde_target_iv_rmse: float = 0.02
    max_pde_mc_iv_rmse: float = 0.015
    max_pde_mass_error: float = 1.0e-8
    max_pde_negative_mass: float = 2.0e-5
    max_abs_mc_forward_z: float = 4.0
    min_market_spread_coverage: float = 0.60

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVMarketAcceptanceCriteria`` instance."""
        for name in (
            "max_dupire_repaired_fraction",
            "max_pde_target_iv_rmse",
            "max_pde_mc_iv_rmse",
            "max_pde_mass_error",
            "max_pde_negative_mass",
            "max_abs_mc_forward_z",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        coverage = float(self.min_market_spread_coverage)
        if not np.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError("min_market_spread_coverage must lie in [0, 1]")
        object.__setattr__(self, "min_market_spread_coverage", coverage)


@dataclass(frozen=True, slots=True)
class LSVMarketPipelineConfig:
    """All model, grid, solver, calibration, MC, and acceptance choices."""

    params: LogSvParams
    calibration_grid: TensorGrid2D
    repricing_grid: TensorGrid2D
    calibration_config: LSVParametricCalibrationConfig
    adi_config: ADIConfig
    dupire_repair_policy: DupireRepairPolicy
    monte_carlo_config: LSVMonteCarloConfig
    calibration_steps_per_interval: tuple[int, ...]
    repricing_steps_per_interval: tuple[int, ...]
    calibration_initial_parameters: FloatArray | None = None
    acceptance_criteria: LSVMarketAcceptanceCriteria = field(
        default_factory=LSVMarketAcceptanceCriteria
    )
    quote_selection: LSVQuoteSelection = LSVQuoteSelection.OTM
    calibration_grid_size: int = 11

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVMarketPipelineConfig`` instance."""
        LogSvCoefficientProvider(self.params)
        for name in ("calibration_grid", "repricing_grid"):
            grid = getattr(self, name)
            if not isinstance(grid, TensorGrid2D):
                raise TypeError(f"{name} must be a TensorGrid2D")
            if not grid.x.nodes[0] < 0.0 < grid.x.nodes[-1]:
                raise ValueError(f"{name} must contain log-forward zero")
            if not grid.y.nodes[0] <= self.params.sigma0 <= grid.y.nodes[-1]:
                raise ValueError(f"{name} must contain sigma0 on its volatility axis")
        if not isinstance(self.calibration_config, LSVParametricCalibrationConfig):
            raise TypeError("calibration_config must be an LSVParametricCalibrationConfig")
        if not isinstance(self.adi_config, ADIConfig):
            raise TypeError("adi_config must be an ADIConfig")
        if not isinstance(self.dupire_repair_policy, DupireRepairPolicy):
            raise TypeError("dupire_repair_policy must be a DupireRepairPolicy")
        if not isinstance(self.monte_carlo_config, LSVMonteCarloConfig):
            raise TypeError("monte_carlo_config must be an LSVMonteCarloConfig")
        if not isinstance(self.acceptance_criteria, LSVMarketAcceptanceCriteria):
            raise TypeError("acceptance_criteria must be LSVMarketAcceptanceCriteria")
        if not isinstance(self.quote_selection, LSVQuoteSelection):
            raise TypeError("quote_selection must be an LSVQuoteSelection")
        for name in (
            "calibration_steps_per_interval",
            "repricing_steps_per_interval",
        ):
            values = _positive_integer_tuple(getattr(self, name), name)
            object.__setattr__(self, name, values)
        if self.calibration_initial_parameters is not None:
            initial = _readonly_matrix(
                self.calibration_initial_parameters,
                "calibration_initial_parameters",
            )
            codec = LocalVolParameterCodec(
                self.calibration_config.family,
                self.calibration_config.formula_weight,
            )
            if initial.shape[1] != codec.size or any(
                not codec.contains(row) for row in initial
            ):
                raise ValueError(
                    "calibration_initial_parameters must contain bounded codec vectors"
                )
            object.__setattr__(self, "calibration_initial_parameters", initial)
        size = int(self.calibration_grid_size)
        if isinstance(self.calibration_grid_size, bool) or size != self.calibration_grid_size:
            raise ValueError("calibration_grid_size must be an integer")
        if size < 5:
            raise ValueError("calibration_grid_size must be at least five")
        object.__setattr__(self, "calibration_grid_size", size)


@dataclass(frozen=True, slots=True)
class LSVPDERepricingSlice:
    """One accepted leverage slice independently replayed on the fine PDE grid."""

    slice_id: str
    maturity: float
    density: DensityState2D
    calls: FloatArray
    implied_volatilities: FloatArray
    diagnostics: tuple[PDEDiagnostics2D, ...]

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVPDERepricingSlice`` instance."""
        maturity = float(self.maturity)
        if not isinstance(self.density, DensityState2D):
            raise TypeError("density must be a DensityState2D")
        if not np.isclose(self.density.time, maturity, rtol=0.0, atol=1.0e-14):
            raise ValueError("density time must match maturity")
        calls = _readonly_vector(self.calls, "calls")
        implied = _readonly_vector(self.implied_volatilities, "implied_volatilities")
        if calls.shape != implied.shape or np.any(calls < 0.0) or np.any(implied <= 0.0):
            raise ValueError("PDE calls and implied volatilities must align and be valid")
        diagnostics = tuple(self.diagnostics)
        if not diagnostics or any(
            not isinstance(item, PDEDiagnostics2D) for item in diagnostics
        ):
            raise ValueError("diagnostics must contain accepted PDE steps")
        object.__setattr__(self, "slice_id", str(self.slice_id))
        object.__setattr__(self, "maturity", maturity)
        object.__setattr__(self, "calls", calls)
        object.__setattr__(self, "implied_volatilities", implied)
        object.__setattr__(self, "diagnostics", diagnostics)


@dataclass(frozen=True, slots=True)
class LSVPDERepricingResult:
    """Sequential fine-grid PDE replay with accepted density carried by maturity."""

    grid: TensorGrid2D
    initial_density: DensityState2D
    slices: tuple[LSVPDERepricingSlice, ...]

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVPDERepricingResult`` instance."""
        if not isinstance(self.grid, TensorGrid2D):
            raise TypeError("grid must be a TensorGrid2D")
        if not isinstance(self.initial_density, DensityState2D):
            raise TypeError("initial_density must be a DensityState2D")
        if self.initial_density.grid is not self.grid:
            raise ValueError("initial_density must use the repricing grid")
        slices = tuple(self.slices)
        if not slices or any(not isinstance(item, LSVPDERepricingSlice) for item in slices):
            raise ValueError("slices must contain LSVPDERepricingSlice objects")
        if any(item.density.grid is not self.grid for item in slices):
            raise ValueError("all replay densities must use the repricing grid")
        object.__setattr__(self, "slices", slices)

    @property
    def model_calls(self) -> FloatArray:
        """Fine-grid model calls in maturity-by-strike order."""
        return _readonly_matrix(np.vstack([item.calls for item in self.slices]), "model_calls")

    @property
    def model_implied_volatilities(self) -> FloatArray:
        """Fine-grid model implied volatilities in maturity-by-strike order."""
        values = np.vstack([item.implied_volatilities for item in self.slices])
        return _readonly_matrix(values, "model_implied_volatilities")


@dataclass(frozen=True, slots=True)
class LSVMonteCarloValidation:
    """Independent normalized-grid MC comparison against fine-grid PDE prices."""

    normalized_pricing: LSVOptionChainPricingResult
    pde_call_differences: FloatArray
    pde_call_z_scores: FloatArray
    pde_iv_differences: FloatArray
    iv_rmse_by_maturity: FloatArray

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVMonteCarloValidation`` instance."""
        if not isinstance(self.normalized_pricing, LSVOptionChainPricingResult):
            raise TypeError("normalized_pricing must be an LSVOptionChainPricingResult")
        differences = _readonly_matrix(self.pde_call_differences, "pde_call_differences")
        z_scores = _readonly_matrix(self.pde_call_z_scores, "pde_call_z_scores")
        iv_differences = _readonly_matrix(self.pde_iv_differences, "pde_iv_differences")
        rmse = _readonly_vector(self.iv_rmse_by_maturity, "iv_rmse_by_maturity")
        expected_shape = (
            len(self.normalized_pricing.slices),
            self.normalized_pricing.slices[0].strikes.size,
        )
        if differences.shape != expected_shape or z_scores.shape != expected_shape:
            raise ValueError("MC price comparison matrices have the wrong shape")
        if iv_differences.shape != expected_shape or rmse.shape != (expected_shape[0],):
            raise ValueError("MC implied-volatility comparison arrays have the wrong shape")
        object.__setattr__(self, "pde_call_differences", differences)
        object.__setattr__(self, "pde_call_z_scores", z_scores)
        object.__setattr__(self, "pde_iv_differences", iv_differences)
        object.__setattr__(self, "iv_rmse_by_maturity", rmse)

    @property
    def forward_z_scores(self) -> FloatArray:
        """Maturity-wise normalized-forward martingale z-scores."""
        values = [item.forward_martingale_z_score for item in self.normalized_pricing.slices]
        return _readonly_vector(values, "forward_z_scores")


@dataclass(frozen=True, slots=True)
class LSVMarketPipelineAcceptance:
    """Observed production metrics and pass/fail status for every configured gate."""

    criteria: LSVMarketAcceptanceCriteria
    dupire_repaired_fraction: float
    pde_target_iv_rmse: FloatArray
    pde_mc_iv_rmse: FloatArray
    pde_mass_errors: FloatArray
    pde_negative_masses: FloatArray
    mc_forward_z_scores: FloatArray
    market_spread_coverage: float
    checks: tuple[tuple[str, bool], ...] = field(init=False)

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVMarketPipelineAcceptance`` instance."""
        if not isinstance(self.criteria, LSVMarketAcceptanceCriteria):
            raise TypeError("criteria must be LSVMarketAcceptanceCriteria")
        repaired = _unit_interval(self.dupire_repaired_fraction, "dupire_repaired_fraction")
        coverage = _unit_interval(self.market_spread_coverage, "market_spread_coverage")
        target_rmse = _readonly_vector(self.pde_target_iv_rmse, "pde_target_iv_rmse")
        mc_rmse = _readonly_vector(self.pde_mc_iv_rmse, "pde_mc_iv_rmse")
        mass_errors = _readonly_vector(self.pde_mass_errors, "pde_mass_errors")
        negative_masses = _readonly_vector(self.pde_negative_masses, "pde_negative_masses")
        forward_z = _readonly_vector(self.mc_forward_z_scores, "mc_forward_z_scores")
        if not (
            target_rmse.shape
            == mc_rmse.shape
            == mass_errors.shape
            == negative_masses.shape
            == forward_z.shape
        ):
            raise ValueError("acceptance maturity metrics must have the same shape")
        criteria = self.criteria
        checks = (
            ("dupire_repair", repaired <= criteria.max_dupire_repaired_fraction),
            ("pde_target_iv", float(np.max(target_rmse)) <= criteria.max_pde_target_iv_rmse),
            ("pde_mc_iv", float(np.max(mc_rmse)) <= criteria.max_pde_mc_iv_rmse),
            ("pde_mass", float(np.max(mass_errors)) <= criteria.max_pde_mass_error),
            (
                "pde_negative_mass",
                float(np.max(negative_masses)) <= criteria.max_pde_negative_mass,
            ),
            (
                "mc_martingale",
                float(np.max(np.abs(forward_z))) <= criteria.max_abs_mc_forward_z,
            ),
            ("market_spread", coverage >= criteria.min_market_spread_coverage),
        )
        object.__setattr__(self, "dupire_repaired_fraction", repaired)
        object.__setattr__(self, "market_spread_coverage", coverage)
        object.__setattr__(self, "pde_target_iv_rmse", target_rmse)
        object.__setattr__(self, "pde_mc_iv_rmse", mc_rmse)
        object.__setattr__(self, "pde_mass_errors", mass_errors)
        object.__setattr__(self, "pde_negative_masses", negative_masses)
        object.__setattr__(self, "mc_forward_z_scores", forward_z)
        object.__setattr__(self, "checks", checks)

    @property
    def passed(self) -> bool:
        """Whether every explicit numerical and data-quality gate passed."""
        return all(passed for _, passed in self.checks)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        """Names of failed acceptance gates."""
        return tuple(name for name, passed in self.checks if not passed)


@dataclass(frozen=True, slots=True)
class LSVMarketPipelineResult:
    """Auditable output of normalization, Dupire, calibration, PDE, and MC."""

    snapshot: FrozenOptionChainSnapshot
    calibration_input: LSVCalibrationInput
    dupire_result: DupireResult
    prefit_local_volatility: LocalVolSurface
    calibration_result: LSVParametricCalibrationResult
    pde_repricing: LSVPDERepricingResult
    monte_carlo_validation: LSVMonteCarloValidation
    option_chain_pricing: LSVOptionChainPricingResult
    acceptance: LSVMarketPipelineAcceptance

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVMarketPipelineResult`` instance."""
        expected_types = (
            ("snapshot", FrozenOptionChainSnapshot),
            ("calibration_input", LSVCalibrationInput),
            ("dupire_result", DupireResult),
            ("prefit_local_volatility", LocalVolSurface),
            ("calibration_result", LSVParametricCalibrationResult),
            ("pde_repricing", LSVPDERepricingResult),
            ("monte_carlo_validation", LSVMonteCarloValidation),
            ("option_chain_pricing", LSVOptionChainPricingResult),
            ("acceptance", LSVMarketPipelineAcceptance),
        )
        for name, expected_type in expected_types:
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"{name} must be {expected_type.__name__}")
        if self.option_chain_pricing.slice_ids != self.snapshot.slice_ids:
            raise ValueError("ragged pricing slices must preserve snapshot identifiers")

    def require_accepted(self) -> None:
        """Raise when one or more configured production gates failed."""
        if not self.acceptance.passed:
            failed = ", ".join(self.acceptance.failed_checks)
            raise RuntimeError(f"frozen LSV pipeline failed acceptance gates: {failed}")

    def audit_summary(self) -> dict[str, str | float | bool]:
        """Return a compact serialization-friendly production audit record."""
        acceptance = self.acceptance
        return {
            "snapshot_id": self.snapshot.snapshot_id,
            "snapshot_fingerprint": self.snapshot.fingerprint,
            "ticker": self.snapshot.ticker,
            "as_of": self.snapshot.as_of,
            "maturities": float(self.snapshot.ttms.size),
            "dupire_repaired_fraction": acceptance.dupire_repaired_fraction,
            "max_pde_target_iv_rmse": float(np.max(acceptance.pde_target_iv_rmse)),
            "max_pde_mc_iv_rmse": float(np.max(acceptance.pde_mc_iv_rmse)),
            "max_pde_mass_error": float(np.max(acceptance.pde_mass_errors)),
            "max_pde_negative_mass": float(np.max(acceptance.pde_negative_masses)),
            "max_abs_mc_forward_z": float(np.max(np.abs(acceptance.mc_forward_z_scores))),
            "market_spread_coverage": acceptance.market_spread_coverage,
            "accepted": acceptance.passed,
        }


@dataclass(frozen=True, slots=True)
class LSVMarketPipeline:
    """Run the frozen-snapshot workflow without changing any numerical kernel."""

    config: LSVMarketPipelineConfig

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVMarketPipeline`` instance."""
        if not isinstance(self.config, LSVMarketPipelineConfig):
            raise TypeError("config must be an LSVMarketPipelineConfig")

    def run(self, snapshot: FrozenOptionChainSnapshot) -> LSVMarketPipelineResult:
        """Normalize, recover Dupire LV, calibrate LSV, and cross-check PDE/MC."""
        if not isinstance(snapshot, FrozenOptionChainSnapshot):
            raise TypeError("snapshot must be a FrozenOptionChainSnapshot")
        chain = snapshot.to_option_chain()
        adapter = LSVOptionChainAdapter(chain, self.config.quote_selection)
        calibration_input = adapter.build_calibration_input(
            grid_size=self.config.calibration_grid_size
        )
        self._validate_run_dimensions(calibration_input)
        call_surface = CallSurface(
            times=calibration_input.target.times,
            strikes=calibration_input.target.strikes,
            calls=calibration_input.target.target_calls,
            forwards=calibration_input.target.forward,
            discount_factors=calibration_input.target.discount_factor,
        )
        dupire = DupireRecoverer().continuous(
            call_surface,
            repair_policy=self.config.dupire_repair_policy,
        )
        local_volatility = LocalVolSurface(
            times=calibration_input.target.times,
            grid=Grid1D(calibration_input.log_moneyness),
            values=dupire.local_volatility,
            coordinate=CoordinateSystem.LOG_FORWARD,
        )
        calibration = LSVParametricCalibrator(
            params=self.config.params,
            grid=self.config.calibration_grid,
            target=calibration_input.target,
            prefit_local_volatility=local_volatility,
            config=self.config.calibration_config,
            adi_config=self.config.adi_config,
        ).calibrate(
            self.config.calibration_steps_per_interval,
            initial_parameters=self.config.calibration_initial_parameters,
        )
        pde_repricing = _reprice_on_fine_grid(
            self.config,
            calibration_input,
            calibration,
        )
        normalized_pricing = _normalized_monte_carlo_pricing(
            self.config,
            calibration_input,
            calibration,
        )
        mc_validation = _monte_carlo_validation(pde_repricing, normalized_pricing)
        chain_pricing = LSVOptionChainPricer(
            LSVMonteCarloPricer(
                self.config.params,
                calibration.leverage_surface,
                self.config.monte_carlo_config,
            )
        ).price(chain)
        acceptance = _evaluate_acceptance(
            self.config.acceptance_criteria,
            calibration_input,
            dupire,
            pde_repricing,
            mc_validation,
        )
        return LSVMarketPipelineResult(
            snapshot=snapshot,
            calibration_input=calibration_input,
            dupire_result=dupire,
            prefit_local_volatility=local_volatility,
            calibration_result=calibration,
            pde_repricing=pde_repricing,
            monte_carlo_validation=mc_validation,
            option_chain_pricing=chain_pricing,
            acceptance=acceptance,
        )

    def _validate_run_dimensions(self, calibration_input: LSVCalibrationInput) -> None:
        """Validate that pipeline outputs align across every maturity and strike slice."""
        maturity_count = calibration_input.target.times.size
        for name in (
            "calibration_steps_per_interval",
            "repricing_steps_per_interval",
        ):
            if len(getattr(self.config, name)) != maturity_count:
                raise ValueError(f"{name} must contain one count per snapshot maturity")
        initial = self.config.calibration_initial_parameters
        if initial is not None and initial.shape[0] != maturity_count:
            raise ValueError(
                "calibration_initial_parameters must contain one row per snapshot maturity"
            )
        log_grid = calibration_input.log_moneyness
        for name in ("calibration_grid", "repricing_grid"):
            x_nodes = getattr(self.config, name).x.nodes
            if log_grid[0] < x_nodes[0] or log_grid[-1] > x_nodes[-1]:
                raise ValueError(f"{name} must contain the normalized market strike domain")


def load_frozen_btc_20211021_snapshot() -> FrozenOptionChainSnapshot:
    """Load the repository's BTC IV snapshot observed on 21 October 2021."""
    from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data

    return FrozenOptionChainSnapshot.from_option_chain(
        get_btc_test_chain_data(),
        snapshot_id="BTC-2021-10-21",
        as_of="2021-10-21",
        source="stochvolmodels.data.sample_option_chains.get_btc_test_chain_data",
    )


def btc_20211021_reference_config() -> LSVMarketPipelineConfig:
    """Return the fully explicit numerical configuration for the BTC pilot."""
    params = LogSvParams(
        sigma0=0.88,
        theta=0.90,
        kappa1=2.0,
        kappa2=3.0,
        beta=-0.25,
        volvol=0.45,
    )
    calibration_grid = TensorGrid2D(
        Grid1D.concentrated(-2.4, 2.2, 71, center=0.0, scale=0.16),
        Grid1D.concentrated(0.05, 2.8, 45, center=0.88, scale=0.10),
    )
    repricing_grid = TensorGrid2D(
        Grid1D.concentrated(-2.4, 2.2, 101, center=0.0, scale=0.13),
        Grid1D.concentrated(0.05, 2.8, 61, center=0.88, scale=0.075),
    )
    calibration_config = LSVParametricCalibrationConfig(
        family=LocalVolFamily.PRODUCT,
        formula_weight=0.2,
        weighting=CalibrationWeighting.VEGA_NORMALIZED,
        maximum_evaluations=12,
        prefit_maximum_evaluations=100,
        optimizer_ftol=5.0e-5,
        optimizer_xtol=5.0e-5,
        optimizer_gtol=5.0e-5,
        leverage_floor=0.10,
        leverage_cap=3.0,
        mass_tolerance=1.0e-6,
        negative_mass_tolerance=2.0e-5,
        fail_on_density_violation=False,
    )
    return LSVMarketPipelineConfig(
        params=params,
        calibration_grid=calibration_grid,
        repricing_grid=repricing_grid,
        calibration_config=calibration_config,
        adi_config=ADIConfig(theta=1.0, scheme=ADIScheme.DOUGLAS),
        dupire_repair_policy=DupireRepairPolicy(
            variance_floor=0.04,
            variance_cap=4.0,
        ),
        monte_carlo_config=LSVMonteCarloConfig(
            paths=100_000,
            steps_per_year=720,
            seed=20260817,
            antithetic=True,
            confidence_level=0.95,
        ),
        calibration_steps_per_interval=(20, 20, 30, 50),
        repricing_steps_per_interval=(80, 80, 100, 160),
        acceptance_criteria=LSVMarketAcceptanceCriteria(),
        quote_selection=LSVQuoteSelection.OTM,
        calibration_grid_size=11,
    )


def _reprice_on_fine_grid(
    config: LSVMarketPipelineConfig,
    calibration_input: LSVCalibrationInput,
    calibration: LSVParametricCalibrationResult,
) -> LSVPDERepricingResult:
    """Reprice calibrated leverage on the configured fine PDE grid."""
    grid = config.repricing_grid
    initial_density = DensityState2D(
        grid,
        grid.project_delta(0.0, config.params.sigma0),
        0.0,
        CoordinateSystem.LOG_FORWARD,
    )
    start_density = initial_density
    criteria = config.acceptance_criteria
    solver = PDE2DADISolver(
        config.adi_config,
        DensityPolicy(
            negativity_tolerance=criteria.max_pde_negative_mass,
            mass_error_tolerance=criteria.max_pde_mass_error,
            renormalize=False,
            fail_on_violation=False,
        ),
    )
    provider = LSVCoefficientProvider(config.params, calibration.leverage_surface)
    boundaries = BoundarySet2D(
        x=BoundaryPair1D.natural(),
        y=BoundaryPair1D.natural(),
    )
    option_types = np.full(calibration_input.target.strikes.size, "C")
    slices: list[LSVPDERepricingSlice] = []
    for slice_id, maturity, step_count in zip(
        calibration_input.slice_ids,
        calibration_input.target.times,
        config.repricing_steps_per_interval,
    ):
        solution = solver.solve(
            grid,
            provider,
            boundaries,
            PDEDirection.FORWARD,
            np.linspace(start_density.time, maturity, step_count + 1),
            start_density.values,
        )
        terminal_density = DensityState2D(
            grid,
            solution.values,
            maturity,
            CoordinateSystem.LOG_FORWARD,
        )
        pricer = DensityPricer(
            terminal_density.marginal_x_state(),
            forward=calibration_input.target.forward,
            discount_factor=calibration_input.target.discount_factor,
        )
        calls = pricer.price_slice(calibration_input.target.strikes).calls
        implied = pricer.implied_volatilities(
            calibration_input.target.strikes,
            option_types,
            ttm=maturity,
        )
        if np.any(~np.isfinite(implied)):
            raise RuntimeError("fine-grid PDE repricing produced non-finite Black IVs")
        slices.append(
            LSVPDERepricingSlice(
                slice_id=slice_id,
                maturity=maturity,
                density=terminal_density,
                calls=calls,
                implied_volatilities=implied,
                diagnostics=solution.diagnostics,
            )
        )
        start_density = terminal_density
    return LSVPDERepricingResult(grid, initial_density, tuple(slices))


def _normalized_monte_carlo_pricing(
    config: LSVMarketPipelineConfig,
    calibration_input: LSVCalibrationInput,
    calibration: LSVParametricCalibrationResult,
) -> LSVOptionChainPricingResult:
    """Price the normalized option chain with the independent LSV simulator."""
    target = calibration_input.target
    normalized_chain = OptionChain(
        ttms=target.times.copy(),
        forwards=np.full(target.times.size, target.forward),
        strikes_ttms=tuple(target.strikes.copy() for _ in target.times),
        optiontypes_ttms=tuple(
            np.full(target.strikes.size, "C") for _ in target.times
        ),
        ids=np.asarray(calibration_input.slice_ids, dtype=str),
        discfactors=np.full(target.times.size, target.discount_factor),
    )
    pricer = LSVMonteCarloPricer(
        config.params,
        calibration.leverage_surface,
        config.monte_carlo_config,
    )
    return LSVOptionChainPricer(pricer).price(normalized_chain)


def _monte_carlo_validation(
    pde: LSVPDERepricingResult,
    normalized_pricing: LSVOptionChainPricingResult,
) -> LSVMonteCarloValidation:
    """Run independent Monte Carlo validation for a calibrated market pipeline."""
    pde_calls = pde.model_calls
    pde_ivs = pde.model_implied_volatilities
    mc_calls = np.vstack([item.model_prices for item in normalized_pricing.slices])
    mc_errors = np.vstack([item.standard_errors for item in normalized_pricing.slices])
    mc_ivs = np.vstack(
        [item.model_implied_volatilities for item in normalized_pricing.slices]
    )
    call_differences = mc_calls - pde_calls
    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = np.divide(
            call_differences,
            mc_errors,
            out=np.zeros_like(call_differences),
            where=mc_errors > 0.0,
        )
    zero_error_difference = (mc_errors == 0.0) & (call_differences != 0.0)
    z_scores[zero_error_difference] = np.copysign(
        np.finfo(np.float64).max,
        call_differences[zero_error_difference],
    )
    iv_differences = mc_ivs - pde_ivs
    iv_rmse = np.sqrt(np.mean(iv_differences * iv_differences, axis=1))
    return LSVMonteCarloValidation(
        normalized_pricing=normalized_pricing,
        pde_call_differences=call_differences,
        pde_call_z_scores=z_scores,
        pde_iv_differences=iv_differences,
        iv_rmse_by_maturity=iv_rmse,
    )


def _evaluate_acceptance(
    criteria: LSVMarketAcceptanceCriteria,
    calibration_input: LSVCalibrationInput,
    dupire: DupireResult,
    pde: LSVPDERepricingResult,
    mc: LSVMonteCarloValidation,
) -> LSVMarketPipelineAcceptance:
    """Evaluate every frozen-market PDE and Monte Carlo acceptance gate."""
    interior_count = int(np.count_nonzero(dupire.interior_mask))
    if interior_count == 0:
        raise RuntimeError("Dupire surface has no interior nodes for acceptance testing")
    repaired_fraction = float(
        np.count_nonzero(dupire.repaired_mask & dupire.interior_mask) / interior_count
    )
    target_iv = calibration_input.target.implied_volatilities
    pde_iv = pde.model_implied_volatilities
    target_rmse = np.sqrt(np.mean((pde_iv - target_iv) ** 2, axis=1))
    mass_errors = np.asarray(
        [abs(item.density.mass - 1.0) for item in pde.slices], dtype=np.float64
    )
    negative_masses = np.asarray(
        [item.density.negative_mass for item in pde.slices], dtype=np.float64
    )
    common_bids = np.vstack(
        [
            np.interp(calibration_input.log_moneyness, coordinates, bids)
            for coordinates, bids in zip(
                calibration_input.source_log_moneyness,
                calibration_input.source_bid_implied_volatilities,
            )
        ]
    )
    common_asks = np.vstack(
        [
            np.interp(calibration_input.log_moneyness, coordinates, asks)
            for coordinates, asks in zip(
                calibration_input.source_log_moneyness,
                calibration_input.source_ask_implied_volatilities,
            )
        ]
    )
    spread_coverage = float(np.mean((pde_iv >= common_bids) & (pde_iv <= common_asks)))
    return LSVMarketPipelineAcceptance(
        criteria=criteria,
        dupire_repaired_fraction=repaired_fraction,
        pde_target_iv_rmse=target_rmse,
        pde_mc_iv_rmse=mc.iv_rmse_by_maturity,
        pde_mass_errors=mass_errors,
        pde_negative_masses=negative_masses,
        mc_forward_z_scores=mc.forward_z_scores,
        market_spread_coverage=spread_coverage,
    )


def _snapshot_fingerprint(snapshot: FrozenOptionChainSnapshot) -> str:
    """Return the deterministic fingerprint of a normalized market snapshot."""
    digest = hashlib.sha256(b"stochvolmodels-frozen-option-chain-v1")
    for value in (
        snapshot.snapshot_id,
        snapshot.as_of,
        snapshot.source,
        snapshot.ticker,
        *snapshot.slice_ids,
    ):
        _update_text_digest(digest, value)
    for values in (
        snapshot.ttms,
        snapshot.forwards,
        snapshot.discount_factors,
        *snapshot.strikes,
        *snapshot.bid_implied_volatilities,
        *snapshot.ask_implied_volatilities,
    ):
        _update_float_digest(digest, values)
    for values in snapshot.option_types:
        for value in values:
            _update_text_digest(digest, str(value))
    return digest.hexdigest()


def _update_text_digest(digest: object, value: str) -> None:
    """Update a fingerprint digest with canonical text values."""
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _update_float_digest(digest: object, values: ArrayLike) -> None:
    """Update a fingerprint digest with canonical floating-point values."""
    array = np.ascontiguousarray(values, dtype="<f8")
    digest.update(array.size.to_bytes(8, "little"))
    digest.update(array.tobytes())


def _positive_integer_tuple(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    """Validate and return strictly positive integer tuple."""
    result = tuple(values)
    if not result or any(
        isinstance(value, bool) or int(value) != value or int(value) < 1 for value in result
    ):
        raise ValueError(f"{name} must contain positive integers")
    return tuple(int(value) for value in result)


def _unit_interval(value: float, name: str) -> float:
    """Validate that a scalar lies in the closed unit interval."""
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _readonly_vector_tuple(
    values: tuple[ArrayLike, ...], name: str
) -> tuple[FloatArray, ...]:
    """Validate vector tuple and return an immutable NumPy value."""
    return tuple(_readonly_vector(value, name) for value in values)


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


def _readonly_strings(values: ArrayLike) -> StringArray:
    """Validate strings and return an immutable NumPy value."""
    array = np.array(values, dtype="U2", copy=True, order="C")
    if array.ndim != 1:
        raise ValueError("option_types must be one-dimensional")
    array.setflags(write=False)
    return array
