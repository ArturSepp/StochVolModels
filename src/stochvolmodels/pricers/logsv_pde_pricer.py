"""Opt-in forward-density pricing for European LogSV calls and puts."""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from numba.typed import List

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.logsv import LogSvMeasure, LogSvModel
from stochvolmodels.models.logsv_pde import LogSvPdeCoefficients
from stochvolmodels.pde_solvers import (
    ADIConfig,
    BoundaryPair1D,
    BoundarySet2D,
    Grid1D,
    PDE2DADISolver,
    PDEDiagnostics2D,
    PDEDirection,
    TensorGrid2D,
)
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.model_pricer import ModelPricer
from stochvolmodels.products.payoffs import _vanilla_payoff
from stochvolmodels.utils.config import OptionType, VariableType


@dataclass(frozen=True, slots=True)
class LogSvPdeExpiry:
    """Expiry diagnostics under the undiscounted law.

    normalized_forward is E[exp(X)] against a market target of one.
    max_parity_error is the largest absolute deviation from D(F-K), in quote currency.
    negative_mass integrates the negative part of the joint density at this expiry.
    """

    ttm: float
    mass: float
    normalized_forward: float
    negative_mass: float
    max_parity_error: float


@dataclass(frozen=True, slots=True)
class LogSvPdeResult:
    """Quote-currency PVs, package Black IVs, failed-inversion masks and diagnostics.

    Each quote array follows the strike order in the input chain. The step diagnostics
    cover every time interval, including intervals between quoted expiries.
    """

    prices: List
    implied_vols: List
    invalid_iv: tuple[np.ndarray, ...]
    expiries: tuple[LogSvPdeExpiry, ...]
    steps: tuple[PDEDiagnostics2D, ...]

    @property
    def peak_negative_mass(self) -> float:
        """Largest joint negative mass over all propagation steps."""
        return max(step.negative_mass for step in self.steps)


def _default_grid(params: LogSvParams, chain: OptionChain) -> TensorGrid2D:
    """Construct an initial mesh containing the initial state and every quoted strike."""
    scale = max(params.theta, params.sigma0)
    stdev = scale * np.sqrt(chain.ttms[-1])
    wing = max(
        np.max(np.abs(np.log(k) - np.log(f))) for k, f in zip(chain.strikes_ttms, chain.forwards)
    )
    width = max(8.0 * stdev, 1.25 * wing)
    x = Grid1D.concentrated(-width, width, 121, 0.0, stdev)
    # Split a single sinh map at sigma0. Both sides then have comparable spacing
    # at the initial delta, even with an asymmetric volatility domain.
    upper = 8.0 * scale
    sigma_scale = 0.25 * params.sigma0
    low_span = np.arcsinh(params.sigma0 / sigma_scale)
    high_span = np.arcsinh((upper - params.sigma0) / sigma_scale)
    n_left = int(np.clip(np.rint(80 * low_span / (low_span + high_span)), 1, 79))
    coords = np.r_[
        np.linspace(-low_span, 0.0, n_left + 1), np.linspace(0.0, high_span, 81 - n_left)[1:]
    ]
    sigma = params.sigma0 + sigma_scale * np.sinh(coords)
    sigma[0], sigma[-1] = 0.0, upper
    return TensorGrid2D(x, Grid1D(sigma))


@dataclass(kw_only=True)
class LogSvPdePricer(ModelPricer):
    """Price ordinary European LogSV vanillas by one forward ADI solve per chain.

    Supply a tensor grid on (normalized log-forward, volatility) to control the
    domain and spatial resolution. steps_per_interval partitions each consecutive
    expiry interval. The starting mesh is a numerical choice and should be refined
    for the maturity, wings and accuracy required.
    """

    grid: TensorGrid2D | None = None
    steps_per_interval: int = 128
    adi: ADIConfig = field(default_factory=lambda: ADIConfig(theta=1.0))

    def _project_chain(
        self,
        option_chain,
        params,
        *,
        measure=LogSvMeasure.MMA,
        variable_type=VariableType.LOG_RETURN,
        is_spot_measure=True,
    ):
        """Solve the undiscounted MMA law and project raw payoffs at exact expiries."""
        if LogSvMeasure(measure) is not LogSvMeasure.MMA or is_spot_measure is not True:
            raise ValueError("LogSvPdePricer supports only the money-market measure (Q_MMA)")
        if variable_type is not VariableType.LOG_RETURN:
            raise ValueError("LogSvPdePricer supports only VariableType.LOG_RETURN")
        if not isinstance(option_chain, OptionChain):
            raise TypeError("option_chain must be an OptionChain")
        if any(not np.all(np.isin(kinds, ["C", "P"])) for kinds in option_chain.optiontypes_ttms):
            raise ValueError("LogSvPdePricer supports only ordinary C and P contracts")
        if (
            isinstance(self.steps_per_interval, bool)
            or not isinstance(self.steps_per_interval, Integral)
            or self.steps_per_interval < 1
        ):
            raise ValueError("steps_per_interval must be a positive integer")

        model = LogSvModel(params)
        provider = LogSvPdeCoefficients(model)
        snapshot = model.params
        grid = self.grid if self.grid is not None else _default_grid(snapshot, option_chain)
        if not isinstance(grid, TensorGrid2D):
            raise TypeError("grid must be a TensorGrid2D")
        if not (
            grid.x.nodes[0] < 0.0 < grid.x.nodes[-1]
            and 0.0 <= grid.y.nodes[0] < snapshot.sigma0 < grid.y.nodes[-1]
        ):
            raise ValueError("initial (0, sigma0) must be interior, with nonnegative volatility")
        maturities = option_chain.ttms
        starts = np.r_[0.0, maturities[:-1]]
        times = np.concatenate(
            (
                [0.0],
                *[
                    np.linspace(a, b, self.steps_per_interval + 1)[1:]
                    for a, b in zip(starts, maturities)
                ],
            )
        )
        # Subintervals shorter than a float ULP can produce repeated partition points.
        times = np.unique(times)
        boundary = BoundaryPair1D.natural()
        law = PDE2DADISolver(self.adi).solve(
            grid,
            provider,
            BoundarySet2D(boundary, boundary),
            PDEDirection.FORWARD,
            times,
            grid.project_delta(0.0, snapshot.sigma0),
            save_times=tuple(maturities),
        )
        relative_spot = np.exp(grid.x.nodes)
        prices, expiries = List(), []
        for i, state in enumerate(law.saved_states):
            weights = grid.marginalize_y(state.values) * grid.x.quadrature_weights
            forward, discount = option_chain.forwards[i], option_chain.discfactors[i]
            strikes, kinds = option_chain.strikes_ttms[i], option_chain.optiontypes_ttms[i]
            spots = forward * relative_spot
            values = [
                discount
                * np.dot(weights, _vanilla_payoff(spots, strike=k, option_type=OptionType(kind)))
                for k, kind in zip(strikes, kinds)
            ]
            prices.append(np.array(values))
            mass = float(weights.sum())
            moment = float(weights @ relative_spot)
            parity = discount * (forward * (moment - 1.0) - strikes * (mass - 1.0))
            expiries.append(
                LogSvPdeExpiry(
                    state.time,
                    mass,
                    moment,
                    -grid.integrate(np.minimum(state.values, 0.0)),
                    float(np.max(np.abs(parity))),
                )
            )
        return prices, tuple(expiries), law.diagnostics

    def price_chain(self, option_chain: OptionChain, params: LogSvParams, **kwargs) -> List:
        """Return discounted raw prices in the ragged input order."""
        prices, _, _ = self._project_chain(option_chain, params, **kwargs)
        return prices

    def price_chain_detailed(
        self, option_chain: OptionChain, params: LogSvParams, **kwargs
    ) -> LogSvPdeResult:
        """Also invert raw prices and expose inversion failures and numerical diagnostics."""
        prices, expiries, steps = self._project_chain(option_chain, params, **kwargs)
        ivols = option_chain.compute_model_ivols_from_chain_data(prices)
        return LogSvPdeResult(prices, ivols, tuple(~np.isfinite(v) for v in ivols), expiries, steps)
