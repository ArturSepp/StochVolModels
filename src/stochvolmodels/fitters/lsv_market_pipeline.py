"""Calibrate LSV leverage to a European OptionChain and assess the fitted vanilla prices.

Examples
--------
Replay three flat Black smiles and inspect the original-unit prices and acceptance checks.

>>> from stochvolmodels.models.local_vol import LocalVolFamily
>>> from stochvolmodels.models.lsv_simulation import LSVMonteCarloConfig
>>> from stochvolmodels.fitters.lsv_parametric import LSVParametricCalibrationConfig
>>> from stochvolmodels.pde_solvers import TensorGrid2D
>>> t = np.array([0.1, 0.2, 0.3])
>>> chain = OptionChain.get_uniform_chain(t, np.array(["1", "2", "3"]),
...                                       strikes=np.linspace(.9, 1.1, 7))
>>> y = Grid1D([.1, .15, .2, .25, .3])
>>> fit_grid = TensorGrid2D(Grid1D.concentrated(-1., 1., 81, 0., .12), y)
>>> replay_grid = TensorGrid2D(Grid1D.concentrated(-1., 1., 121, 0., .12), y)
>>> config = LSVParametricCalibrationConfig(family=LocalVolFamily.FLAT)
>>> fitter = LSVParametricCalibrator(fit_grid, config)
>>> pipeline = LSVMarketPipeline(fitter, LogSvPdePricer(grid=replay_grid, steps_per_interval=32),
...                              LSVMonteCarloPricer(config=LSVMonteCarloConfig(8192, 120, 42)))
>>> params = LogSvParams(sigma0=.2, theta=.2, beta=0., volvol=0.)
>>> result = pipeline.run(chain, params, grid_size=7, repair_policy=DupireRepairPolicy(.01, .16))
>>> len(result.original), len(result.original[0].pde_prices), len(result.acceptance.checks)
(3, 7, 7)
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.fitters.dupire import DupireRepairPolicy, DupireResult
from stochvolmodels.fitters.lsv_market import (
    LSVCalibrationInput,
    LSVOptionChainAdapter,
    LSVQuoteSelection,
)
from stochvolmodels.fitters.lsv_parametric import (
    LSVParametricCalibrationResult,
    LSVParametricCalibrator,
)
from stochvolmodels.models.local_vol import LocalVolSurface
from stochvolmodels.pde_solvers import Grid1D, PDEDiagnostics2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdeExpiry, LogSvPdePricer, LogSvPdeResult
from stochvolmodels.pricers.lsv_mc_pricer import LSVMonteCarloPricer, LSVMonteCarloResult

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LSVMarketAcceptanceCriteria:
    """Inclusive limits on the independent replay. IV errors are decimal annualized Black IV."""

    max_dupire_repaired_fraction: float = 0.15
    max_pde_target_iv_rmse: float = 0.02
    max_pde_mc_iv_rmse: float = 0.015
    max_pde_mass_error: float = 1.0e-8
    max_pde_negative_mass: float = 2.0e-5
    max_abs_mc_forward_z: float = 4.0
    min_market_spread_coverage: float = 0.60

    def __post_init__(self) -> None:
        """Require nonnegative finite thresholds and fractions in the unit interval."""
        for spec in fields(self):
            value = float(getattr(self, spec.name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{spec.name} must be finite and nonnegative")
            object.__setattr__(self, spec.name, value)
        if max(self.max_dupire_repaired_fraction, self.min_market_spread_coverage) > 1:
            raise ValueError("repair and coverage fractions must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class LSVMarketAcceptance:
    """Common-grid diagnostics. Per-maturity values follow the source expiry order."""

    criteria: LSVMarketAcceptanceCriteria
    dupire_repaired_fraction: float
    pde_target_iv_rmse: tuple[float, ...]
    pde_mc_iv_rmse: tuple[float, ...]
    pde_mass_errors: tuple[float, ...]
    pde_negative_masses: tuple[float, ...]
    mc_forward_z_scores: tuple[float, ...]
    market_spread_coverage: float

    @property
    def checks(self) -> tuple[tuple[str, bool], ...]:
        """Evaluate each limit across all its observations, including failed IV inversions."""
        c = self.criteria
        return (
            ("dupire_repair", self.dupire_repaired_fraction <= c.max_dupire_repaired_fraction),
            (
                "pde_target_iv",
                bool(np.max(self.pde_target_iv_rmse) <= c.max_pde_target_iv_rmse),
            ),
            ("pde_mc_iv", bool(np.max(self.pde_mc_iv_rmse) <= c.max_pde_mc_iv_rmse)),
            ("pde_mass", bool(np.max(self.pde_mass_errors) <= c.max_pde_mass_error)),
            (
                "pde_negative_mass",
                bool(np.max(self.pde_negative_masses) <= c.max_pde_negative_mass),
            ),
            (
                "mc_martingale",
                bool(np.max(np.abs(self.mc_forward_z_scores)) <= c.max_abs_mc_forward_z),
            ),
            ("market_spread", self.market_spread_coverage >= c.min_market_spread_coverage),
        )

    @property
    def failed_checks(self) -> tuple[str, ...]:
        """Names of failed checks in the historical evaluation order."""
        return tuple(name for name, passed in self.checks if not passed)

    @property
    def passed(self) -> bool:
        """Whether all seven final checks passed."""
        return not self.failed_checks


@dataclass(frozen=True, slots=True)
class LSVMarketPriceSlice:
    """PDE and MC vanilla PVs, MC standard errors and raw-price Black IVs at one expiry."""

    pde_prices: FloatArray
    mc_prices: FloatArray
    mc_standard_errors: FloatArray
    pde_implied_vols: FloatArray
    mc_implied_vols: FloatArray


@dataclass(frozen=True, slots=True)
class LSVMarketPipelineResult:
    """Market data, fitted model, and independent prices in normalized and original units."""

    market: LSVCalibrationInput
    dupire: DupireResult
    prefit_local_volatility: LocalVolSurface
    calibration: LSVParametricCalibrationResult
    normalized: tuple[LSVMarketPriceSlice, ...]
    original: tuple[LSVMarketPriceSlice, ...]
    pde_expiries: tuple[LogSvPdeExpiry, ...]
    pde_steps: tuple[PDEDiagnostics2D, ...]
    acceptance: LSVMarketAcceptance

    def require_accepted(self) -> None:
        """Raise with every failed final check when the caller requires an accepted fit."""
        if not self.acceptance.passed:
            raise RuntimeError(
                "LSV market calibration failed: " + ", ".join(self.acceptance.failed_checks)
            )


@dataclass(frozen=True, slots=True)
class LSVMarketPipeline:
    """Fit leverage at fixed LogSV dynamics and replay the fitted model by PDE and MC.

    Supply the fit grid via the calibrator and the replay grid and step counts via
    the PDE pricer. Candidate-density limits belong to the calibrator. Final checks
    use common-grid calls, observed-expiry mass and an independent MC forward estimate.
    """

    calibrator: LSVParametricCalibrator
    pde_pricer: LogSvPdePricer
    mc_pricer: LSVMonteCarloPricer = field(default_factory=LSVMonteCarloPricer)
    criteria: LSVMarketAcceptanceCriteria = field(default_factory=LSVMarketAcceptanceCriteria)

    def __post_init__(self) -> None:
        """Validate the fit, replay engines and acceptance policy before running an optimizer."""
        if not isinstance(self.calibrator, LSVParametricCalibrator):
            raise TypeError("calibrator must be an LSVParametricCalibrator")
        if not isinstance(self.pde_pricer, LogSvPdePricer):
            raise TypeError("pde_pricer must be a LogSvPdePricer")
        if not isinstance(self.mc_pricer, LSVMonteCarloPricer):
            raise TypeError("mc_pricer must be an LSVMonteCarloPricer")
        if not isinstance(self.criteria, LSVMarketAcceptanceCriteria):
            raise TypeError("criteria must be LSVMarketAcceptanceCriteria")

    def run(
        self,
        option_chain: OptionChain,
        params: LogSvParams,
        *,
        grid_size: int = 11,
        quote_selection: LSVQuoteSelection = LSVQuoteSelection.OTM,
        repair_policy: DupireRepairPolicy | None = None,
        historical_dupire_endpoints: bool = False,
        calibration_steps_per_interval: int | ArrayLike = 20,
        initial_parameters: ArrayLike | None = None,
    ) -> LSVMarketPipelineResult:
        """Snapshot quotes, recover Dupire variance, fit leverage and evaluate all final checks.

        Original PVs and standard errors are in quote currency. Common-grid calls and
        errors are divided by the expiry's discounted forward. Failed Black inversions
        retain NaN IVs. historical_dupire_endpoints selects the 2.18.0 prefit for replication.
        A failed calibration trial raises before final validation.
        """
        market = LSVOptionChainAdapter(option_chain, quote_selection).build_calibration_input(
            grid_size=grid_size
        )
        for grid in (self.calibrator.grid, self.pde_pricer.grid):
            if grid is not None and (
                market.log_moneyness[0] < grid.x.nodes[0]
                or market.log_moneyness[-1] > grid.x.nodes[-1]
            ):
                raise ValueError(
                    "the calibration and repricing grids must contain the common market strikes"
                )
        dupire = market.recover_dupire(repair_policy=repair_policy)
        buckets = dupire.to_local_vol_surface(historical_endpoints=historical_dupire_endpoints)
        # The normalized strikes are exp(log_moneyness). Restore their exact quoted
        # coordinates without interpolating through log(exp(x)) in the optimizer start.
        local = LocalVolSurface(buckets.times, Grid1D(market.log_moneyness), buckets.values)
        fit = self.calibrator.calibrate(
            params,
            market.target,
            local,
            calibration_steps_per_interval,
            initial_parameters=initial_parameters,
        )
        # Both quote sets use the same state at each expiry. Append common-grid calls
        # at actual strikes so one PDE propagation and one simulation price all quotes.
        rows = market.source_slices
        call_types = np.full(market.target.strikes.size, "C")
        batch = OptionChain(
            ttms=market.target.times,
            forwards=np.array([row.forward for row in rows]),
            discfactors=np.array([row.discount_factor for row in rows]),
            ids=np.array([row.slice_id for row in rows]),
            strikes_ttms=tuple(
                np.r_[row.strikes, row.forward * market.target.strikes] for row in rows
            ),
            optiontypes_ttms=tuple(np.concatenate((row.option_types, call_types)) for row in rows),
        )
        pde = self.pde_pricer.price_chain_detailed(batch, fit.model)
        mc = self.mc_pricer.price_chain_detailed(batch, fit.model)
        original, normalized = [], []
        forward_z = []
        for i, row in enumerate(rows):
            n = len(row.strikes)
            original.append(_price_slice(pde, mc, i, slice(0, n), 1.0))
            normalized.append(
                _price_slice(pde, mc, i, slice(n, None), row.forward * row.discount_factor)
            )
            expiry = mc.expiries[i]
            difference = expiry.forward_mean - row.forward
            if expiry.forward_standard_error == 0:
                z = 0.0 if difference == 0 else np.copysign(np.inf, difference)
            else:
                z = difference / expiry.forward_standard_error
            forward_z.append(float(z))
        pde_iv = np.vstack([s.pde_implied_vols for s in normalized])
        mc_iv = np.vstack([s.mc_implied_vols for s in normalized])
        acceptance = LSVMarketAcceptance(
            criteria=self.criteria,
            dupire_repaired_fraction=float(np.mean(dupire.repaired_mask[dupire.interior_mask])),
            pde_target_iv_rmse=tuple(
                np.sqrt(np.mean((pde_iv - market.target.implied_volatilities) ** 2, axis=1))
            ),
            pde_mc_iv_rmse=tuple(np.sqrt(np.mean((pde_iv - mc_iv) ** 2, axis=1))),
            pde_mass_errors=tuple(abs(s.mass - 1.0) for s in pde.expiries),
            pde_negative_masses=tuple(s.negative_mass for s in pde.expiries),
            mc_forward_z_scores=tuple(forward_z),
            market_spread_coverage=float(
                np.mean((market.bid_ivs <= pde_iv) & (pde_iv <= market.ask_ivs))
            ),
        )
        return LSVMarketPipelineResult(
            market,
            dupire,
            local,
            fit,
            tuple(normalized),
            tuple(original),
            pde.expiries,
            pde.steps,
            acceptance,
        )


def _price_slice(
    pde: LogSvPdeResult, mc: LSVMonteCarloResult, i: int, indices: slice, scale: float
) -> LSVMarketPriceSlice:
    """Extract one quote set, scaling PVs and their sampling errors in the same units."""
    values = LSVMarketPriceSlice(
        pde.prices[i][indices] / scale,
        mc.prices[i][indices] / scale,
        mc.standard_errors[i][indices] / scale,
        pde.implied_vols[i][indices].copy(),
        mc.implied_vols[i][indices].copy(),
    )
    for spec in fields(values):
        getattr(values, spec.name).setflags(write=False)
    return values
