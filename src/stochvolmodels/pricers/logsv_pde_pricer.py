"""Opt-in forward-density pricing for European LogSV calls and puts."""

from dataclasses import dataclass, field, replace
from numbers import Integral

import numpy as np
from numba.typed import List
from numpy.typing import ArrayLike
from scipy.stats import poisson

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.logsv import LogSvMeasure, LogSvModel
from stochvolmodels.models.lsv import FixedLogForwardJump, LSVModel
from stochvolmodels.models.logsv_pde import LogSvPdeCoefficients
from stochvolmodels.pde_solvers import (
    ADIConfig,
    BoundaryMass2D,
    BoundaryPair1D,
    BoundarySet2D,
    Grid1D,
    PDE2DADISolver,
    PDEDiagnostics2D,
    PDEDirection,
    PDEState2D,
    TensorGrid2D,
)
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.lsv_projection import _advance_lsv, _interval_time_grid, _step_counts
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
    boundary_mass: BoundaryMass2D | None = None


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


def _default_grid(
    params: LogSvParams,
    chain: OptionChain,
    return_loading: float = 1.0,
    jump: FixedLogForwardJump | None = None,
    *,
    first_return_loading: float | None = None,
) -> TensorGrid2D:
    """Construct an initial mesh containing the initial state and every quoted strike."""
    scale = max(params.theta, params.sigma0)
    if first_return_loading is None:
        first_return_loading = return_loading
    first_stdev = first_return_loading * scale * np.sqrt(chain.ttms[0])
    last_stdev = return_loading * scale * np.sqrt(chain.ttms[-1])
    wing = max(
        np.max(np.abs(np.log(k) - np.log(f))) for k, f in zip(chain.strikes_ttms, chain.forwards)
    )
    width = max(8.0 * last_stdev, 1.25 * wing)
    lower_x, upper_x = -width, width
    if jump is not None and jump.intensity > 0 and jump.jump_size != 0:
        maturity = chain.ttms[-1]
        # Cover count tails under both the probability and forward-moment weights.
        count_mean = maturity * max(jump.intensity, jump.intensity - jump.compensator)
        count = max(1.0, poisson.isf(1e-10, count_mean))
        drift = maturity * jump.compensator
        displacement = count * jump.jump_size
        lower_x += min(0.0, drift) + min(0.0, displacement)
        upper_x += max(0.0, drift) + max(0.0, displacement)
    # Keep the one-expiry spacing in sinh coordinates as the front-expiry law narrows.
    # An odd number of nodes keeps zero on the mesh for a symmetric log-forward domain.
    bounds = np.array([-lower_x, upper_x])
    first_span = np.arcsinh(bounds / first_stdev).sum()
    last_span = np.arcsinh(bounds / last_stdev).sum()
    half_intervals = int(np.ceil(60 * (first_span / last_span)))
    x = Grid1D.concentrated(lower_x, upper_x, 2 * half_intervals + 1, 0.0, first_stdev)
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
    expiry interval, with either one count or a count per expiry. Refine the mesh
    for the maturity, wings and accuracy required. Pass an LSVModel to use a
    leverage surface with optional backbone. Each leverage or backbone knot also
    splits the integration interval and receives steps_per_interval steps. The first
    bucket uses quadratic time spacing from the initial point mass. Later buckets use
    uniform spacing. Set quadratic_start=False for uniform spacing throughout.
    Fixed log-forward jumps use compensated drift and symmetric compound-Poisson
    splitting. Each step reports positive mass translated outside the x grid.
    """

    grid: TensorGrid2D | None = None
    steps_per_interval: int | ArrayLike = 128
    adi: ADIConfig = field(default_factory=lambda: ADIConfig(theta=1.0))
    quadratic_start: bool = True

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
        if np.ndim(self.steps_per_interval) == 0 and (
            isinstance(self.steps_per_interval, bool)
            or not isinstance(self.steps_per_interval, Integral)
        ):
            raise ValueError("steps_per_interval must be a positive integer")
        counts = _step_counts(self.steps_per_interval, len(option_chain.ttms))

        lsv = isinstance(params, LSVModel)
        model = params.dynamics if lsv else LogSvModel(params)
        snapshot = model.params
        grid = self.grid
        if grid is None:
            first_loading = loading = 1.0
            if lsv:
                first, last = option_chain.ttms[0], option_chain.ttms[-1]
                knots = params.time_knots
                ends = np.union1d(knots[(knots > 0) & (knots < last)], [first, last])
                loadings = np.array([params.max_return_loading(t) for t in ends])
                first_loading = loadings[ends <= first].max()
                loading = loadings.max()
            grid = _default_grid(
                snapshot, option_chain, loading, params.jump if lsv else None,
                first_return_loading=first_loading,
            )
        if not isinstance(grid, TensorGrid2D):
            raise TypeError("grid must be a TensorGrid2D")
        if not (
            grid.x.nodes[0] < 0.0 < grid.x.nodes[-1]
            and 0.0 <= grid.y.nodes[0] < snapshot.sigma0 < grid.y.nodes[-1]
        ):
            raise ValueError("initial (0, sigma0) must be interior, with nonnegative volatility")
        maturities = option_chain.ttms
        solver = PDE2DADISolver(self.adi)
        start_values = grid.project_delta(0.0, snapshot.sigma0)
        if lsv:
            state = PDEState2D(0.0, start_values)
            states, diagnostics = [], []
            for end, count in zip(maturities, counts):
                state, steps = _advance_lsv(
                    params, grid, solver, state, end, count, quadratic_start=self.quadratic_start
                )
                states.append(state)
                diagnostics.extend(steps)
            steps = tuple(replace(d, step_index=i) for i, d in enumerate(diagnostics))
        else:
            starts = np.r_[0.0, maturities[:-1]]
            times = np.concatenate(
                (
                    [0.0],
                    *[
                        _interval_time_grid(a, b, count, self.quadratic_start)[1:]
                        for a, b, count in zip(starts, maturities, counts)
                    ],
                )
            )
            boundary = BoundaryPair1D.natural()
            law = solver.solve(
                grid,
                LogSvPdeCoefficients(model),
                BoundarySet2D(boundary, boundary),
                PDEDirection.FORWARD,
                times,
                start_values,
                save_times=tuple(maturities),
            )
            states, steps = law.saved_states, law.diagnostics
        relative_spot = np.exp(grid.x.nodes)
        prices, expiries = List(), []
        for i, state in enumerate(states):
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
                    grid.boundary_mass(state.values),
                )
            )
        return prices, tuple(expiries), steps

    def price_chain(
        self, option_chain: OptionChain, params: LogSvParams | LSVModel, **kwargs
    ) -> List:
        """Return discounted raw prices in the ragged input order."""
        prices, _, _ = self._project_chain(option_chain, params, **kwargs)
        return prices

    def price_chain_detailed(
        self, option_chain: OptionChain, params: LogSvParams | LSVModel, **kwargs
    ) -> LogSvPdeResult:
        """Also invert raw prices and expose inversion failures and numerical diagnostics."""
        prices, expiries, steps = self._project_chain(option_chain, params, **kwargs)
        ivols = option_chain.compute_model_ivols_from_chain_data(prices)
        return LogSvPdeResult(prices, ivols, tuple(~np.isfinite(v) for v in ivols), expiries, steps)
