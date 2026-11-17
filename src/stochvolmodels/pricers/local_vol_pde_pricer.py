"""Opt-in one-dimensional forward PDE valuation of deterministic local-volatility vanillas.

Examples
--------
Price a six-month call using two distinct volatility buckets.

>>> from stochvolmodels.models.local_vol import LocalVolSurface
>>> from stochvolmodels.pde_solvers import Grid1D
>>> from stochvolmodels.pricers.local_vol_pde_pricer import LocalVolPdePricer
>>> axis = Grid1D.uniform(-0.5, 0.5, 3)
>>> surface = LocalVolSurface([0.25, 1.0], axis, [[0.24, 0.20, 0.18], [0.26, 0.21, 0.19]])
>>> price, iv = LocalVolPdePricer().price_vanilla(surface, 0.5, 100., 100., "C", 0.98)
>>> bool(5. < price < 7. and 0.19 < iv < 0.22)
True
"""

from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
from numba.typed import List

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.local_vol import (
    CoordinateSystem,
    LocalVolCoefficientProvider1D,
    LocalVolSurface,
)
from stochvolmodels.pde_solvers import (
    BoundaryMass1D,
    BoundaryPair1D,
    Grid1D,
    PDE1DSolver,
    PDEDiagnostics,
    PDEDirection,
    StaticCoefficientProvider1D,
    TimeStepConfig,
)
from stochvolmodels.pricers.model_pricer import ModelPricer
from stochvolmodels.products.payoffs import _vanilla_payoff
from stochvolmodels.utils.config import OptionType, VariableType


@dataclass(frozen=True, slots=True)
class LocalVolPdeExpiry:
    """Diagnostics under the raw undiscounted law at a quoted maturity."""

    ttm: float
    mass: float
    normalized_forward: float
    negative_mass: float
    max_parity_error: float
    boundary_mass: BoundaryMass1D | None = None


@dataclass(frozen=True, slots=True)
class LocalVolPdeResult:
    """Ragged quote-currency PVs, Black IVs and numerical diagnostics.

    normalized_forward has target one. max_parity_error is measured against the
    input discounted forward-minus-strike, in the option's quote currency.
    steps contains the raw diagnostics in chronological order, including startup
    half-steps. peak_negative_mass is the largest negative mass over these steps.
    """

    prices: List
    implied_vols: List
    invalid_iv: tuple[np.ndarray, ...]
    expiries: tuple[LocalVolPdeExpiry, ...]
    peak_negative_mass: float
    steps: tuple[PDEDiagnostics, ...] = ()


def _default_grid(chain: OptionChain, surface: LocalVolSurface) -> Grid1D:
    """Construct a log-spaced tail domain around the unit forward or initial level."""
    levels = np.concatenate(
        [np.log(k) - np.log(f) for k, f in zip(chain.strikes_ttms, chain.forwards)]
    )
    # Bound integrated variance by the largest volatility in each active bucket.
    ends = np.r_[
        surface.times[(surface.times > 0) & (surface.times < chain.ttms[-1])], chain.ttms[-1]
    ]
    rows = np.minimum(np.searchsorted(surface.times, ends), len(surface.times) - 1)
    sd = np.sqrt(np.dot(np.diff(np.r_[0.0, ends]), np.max(surface.values[rows], axis=1) ** 2))
    width = max(8 * sd, 1.25 * np.max(np.abs(levels)), 1e-4)
    x = Grid1D.concentrated(-width, width, 401, 0.0, max(sd, 1e-5))
    if surface.coordinate is CoordinateSystem.LOG_FORWARD:
        return x
    return Grid1D(chain.forwards[0] * np.exp(x.nodes))


@dataclass(kw_only=True)
class LocalVolPdePricer(ModelPricer):
    """Price European calls and puts under a supplied deterministic local-volatility surface.

    The optional solver grid uses the surface's coordinate and is independent of
    its interpolation nodes. Each expiry or interior bucket end splits the time
    grid. steps_per_interval applies to each resulting interval. By default
    (theta=None), two implicit Euler half-steps replace the first full step of
    each interval, then Crank–Nicolson advances the remaining steps. A numeric
    theta applies throughout, including the first step (1 for implicit Euler,
    0.5 for plain Crank–Nicolson). A LEVEL law has one constant forward.
    """

    grid: Grid1D | None = None
    steps_per_interval: int = 128
    theta: float | None = None

    def _project_chain(
        self, option_chain, params, *, variable_type=VariableType.LOG_RETURN, is_spot_measure=True,
        collect_steps=False,
    ):
        """Propagate the unit point mass once through all buckets and project raw payoffs."""
        if not isinstance(option_chain, OptionChain) or not isinstance(params, LocalVolSurface):
            raise TypeError("an OptionChain and a LocalVolSurface are required")
        if variable_type is not VariableType.LOG_RETURN or is_spot_measure is not True:
            raise ValueError(
                "only ordinary European vanillas under the money-market measure are supported"
            )
        if any(not np.isin(kinds, ["C", "P"]).all() for kinds in option_chain.optiontypes_ttms):
            raise ValueError("only ordinary C and P contracts are supported")
        if (
            isinstance(self.steps_per_interval, bool)
            or not isinstance(self.steps_per_interval, Integral)
            or self.steps_per_interval < 1
        ):
            raise ValueError("steps_per_interval must be a positive integer")

        relative = params.coordinate is CoordinateSystem.LOG_FORWARD
        start_value = 0.0 if relative else option_chain.forwards[0]
        if not relative and not np.all(option_chain.forwards == start_value):
            raise ValueError("absolute level dynamics require a constant forward across expiries")
        grid = self.grid if self.grid is not None else _default_grid(option_chain, params)
        if not isinstance(grid, Grid1D):
            raise TypeError("grid must be a Grid1D")
        if not grid.nodes[0] < start_value < grid.nodes[-1]:
            raise ValueError("the initial state must be inside the solver grid")

        maturities = option_chain.ttms
        knots = params.times[(params.times > 0) & (params.times < maturities[-1])]
        ends = np.union1d(maturities, knots)
        provider = LocalVolCoefficientProvider1D(params)
        solver = PDE1DSolver(TimeStepConfig(theta=0.5 if self.theta is None else self.theta))
        startup = PDE1DSolver(TimeStepConfig(theta=1.0)) if self.theta is None else None
        boundaries = BoundaryPair1D.natural()
        state = grid.project_delta(start_value)
        nodes = np.exp(grid.nodes) if relative else grid.nodes
        prices, expiries = List(), []
        steps = []
        peak, start, i = 0.0, 0.0, 0
        for end in ends:
            # Inclusive right endpoint selects the coefficient valid on (start, end].
            fixed = StaticCoefficientProvider1D(provider.evaluate(end, grid))
            times = np.unique(np.linspace(start, end, self.steps_per_interval + 1))
            segments = []
            if startup is not None:
                # A volatility or step-size jump can excite oscillations again at a knot.
                mid = start + 0.5 * (times[1] - start)
                segments.append((startup, np.unique([start, mid, times[1]])))
                times = times[1:]
            if times.size > 1:
                segments.append((solver, times))
            for integrator, mesh in segments:
                law = integrator.solve(grid, fixed, boundaries, PDEDirection.FORWARD, mesh, state)
                state = law.values
                peak = max(peak, max(d.negative_mass for d in law.diagnostics))
                if collect_steps:
                    offset = len(steps)
                    steps.extend(
                        replace(d, step_index=offset + d.step_index) for d in law.diagnostics
                    )
            if end == maturities[i]:
                forward = option_chain.forwards[i]
                discount = option_chain.discfactors[i]
                strikes = option_chain.strikes_ttms[i]
                underlying = forward * nodes if relative else nodes
                weights = grid.quadrature_weights * state
                prices.append(
                    np.array(
                        [
                            discount
                            * np.dot(
                                weights,
                                _vanilla_payoff(underlying, strike=k, option_type=OptionType(kind)),
                            )
                            for k, kind in zip(strikes, option_chain.optiontypes_ttms[i])
                        ]
                    )
                )
                mass = weights.sum()
                moment = weights @ underlying
                parity_gap = discount * ((moment - forward) - strikes * (mass - 1.0))
                expiries.append(
                    LocalVolPdeExpiry(
                        float(end),
                        float(mass),
                        float(moment / forward),
                        law.diagnostics[-1].negative_mass,
                        float(np.max(np.abs(parity_gap))),
                        grid.boundary_mass(state),
                    )
                )
                i += 1
            start = end
        return prices, tuple(expiries), peak, tuple(steps)

    def price_chain(self, option_chain: OptionChain, params: LocalVolSurface, **kwargs) -> List:
        """Return discounted vanilla premiums in original expiry and strike order."""
        prices, _, _, _ = self._project_chain(option_chain, params, **kwargs)
        return prices

    def price_chain_detailed(
        self, option_chain: OptionChain, params: LocalVolSurface, **kwargs
    ) -> LocalVolPdeResult:
        """Also invert raw premiums and return expiry and individual time-step diagnostics."""
        prices, expiries, peak, steps = self._project_chain(
            option_chain, params, collect_steps=True, **kwargs
        )
        ivs = option_chain.compute_model_ivols_from_chain_data(prices)
        return LocalVolPdeResult(
            prices, ivs, tuple(~np.isfinite(v) for v in ivs), expiries, peak, steps
        )
