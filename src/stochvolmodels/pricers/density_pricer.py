"""
Immutable density states and European-vanilla pricing projections.

Density states retain their coordinate convention, quadrature grid, and immutable values.
Pricing projections integrate calls and puts directly from one- or two-dimensional states, while
implied-volatility conversion delegates to the shared vanilla-pricer dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.models.local_vol import CoordinateSystem
from stochvolmodels.pde_solvers.grids import Grid1D, TensorGrid2D
from stochvolmodels.utils._arrays import readonly_float_array as _readonly_array

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DensityState1D:
    """Pointwise one-dimensional density under the grid quadrature convention."""

    grid: Grid1D
    values: FloatArray
    time: float
    coordinate: CoordinateSystem

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``DensityState1D`` instance."""
        if not isinstance(self.grid, Grid1D):
            raise TypeError("grid must be a Grid1D")
        values = _readonly_array(
            self.grid.validate_shape(self.values), name="density values", ndim=1
        )
        time = float(self.time)
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("time must be finite and non-negative")
        if not isinstance(self.coordinate, CoordinateSystem):
            raise TypeError("coordinate must be a CoordinateSystem")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "time", time)

    @property
    def mass(self) -> float:
        """Weighted density mass."""
        return self.grid.integrate(self.values)

    @property
    def negative_mass(self) -> float:
        """Weighted integral of the negative part, reported as a positive number."""
        return -self.grid.integrate(np.minimum(self.values, 0.0))

    @property
    def coordinate_first_moment(self) -> float:
        """First moment in the stored coordinate."""
        return self.grid.integrate(self.grid.nodes * self.values)


@dataclass(frozen=True, slots=True)
class DensityState2D:
    """Pointwise tensor density whose x axis is the pricing coordinate."""

    grid: TensorGrid2D
    values: FloatArray
    time: float
    x_coordinate: CoordinateSystem

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``DensityState2D`` instance."""
        if not isinstance(self.grid, TensorGrid2D):
            raise TypeError("grid must be a TensorGrid2D")
        values = _readonly_array(
            self.grid.validate_shape(self.values), name="density values", ndim=2
        )
        time = float(self.time)
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("time must be finite and non-negative")
        if not isinstance(self.x_coordinate, CoordinateSystem):
            raise TypeError("x_coordinate must be a CoordinateSystem")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "time", time)

    @property
    def mass(self) -> float:
        """Weighted density mass."""
        return self.grid.integrate(self.values)

    @property
    def negative_mass(self) -> float:
        """Weighted integral of the negative part, reported as a positive number."""
        return -self.grid.integrate(np.minimum(self.values, 0.0))

    def marginal_x(self) -> FloatArray:
        """Integrate out y and return pointwise density on the x grid."""
        return self.grid.marginalize_y(self.values)

    def marginal_y(self) -> FloatArray:
        """Integrate out x and return pointwise density on the y grid."""
        return self.grid.marginalize_x(self.values)

    def marginal_x_state(self) -> DensityState1D:
        """Return an immutable pricing-axis marginal state."""
        return DensityState1D(
            grid=self.grid.x,
            values=self.marginal_x(),
            time=self.time,
            coordinate=self.x_coordinate,
        )


@dataclass(frozen=True, slots=True)
class DensityPricingResult:
    """Calls, puts, and density-consistent parity diagnostics for one strike slice."""

    strikes: FloatArray
    calls: FloatArray
    puts: FloatArray
    parity_residuals: FloatArray
    mass: float
    forward_moment: float
    discount_factor: float

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``DensityPricingResult`` instance."""
        strikes = _readonly_array(self.strikes, name="strikes", ndim=1)
        calls = _readonly_array(self.calls, name="calls", ndim=1)
        puts = _readonly_array(self.puts, name="puts", ndim=1)
        residuals = _readonly_array(
            self.parity_residuals, name="parity_residuals", ndim=1
        )
        if not (strikes.shape == calls.shape == puts.shape == residuals.shape):
            raise ValueError("pricing result arrays must have the same shape")
        object.__setattr__(self, "strikes", strikes)
        object.__setattr__(self, "calls", calls)
        object.__setattr__(self, "puts", puts)
        object.__setattr__(self, "parity_residuals", residuals)
        object.__setattr__(self, "mass", float(self.mass))
        object.__setattr__(self, "forward_moment", float(self.forward_moment))
        object.__setattr__(self, "discount_factor", float(self.discount_factor))


@dataclass(frozen=True, slots=True)
class DensityPricer:
    """European-vanilla pricing adapter for a one-dimensional terminal density."""

    state: DensityState1D
    forward: float
    discount_factor: float = 1.0

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``DensityPricer`` instance."""
        if not isinstance(self.state, DensityState1D):
            raise TypeError("state must be a DensityState1D")
        forward = float(self.forward)
        discount = float(self.discount_factor)
        if not np.isfinite(forward) or forward <= 0.0:
            raise ValueError("forward must be finite and positive")
        if not np.isfinite(discount) or discount <= 0.0:
            raise ValueError("discount_factor must be finite and positive")
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "discount_factor", discount)

    @property
    def underlying_nodes(self) -> FloatArray:
        """Terminal underlying represented at each density node."""
        if self.state.coordinate is CoordinateSystem.LOG_FORWARD:
            return np.asarray(
                self.forward * np.exp(self.state.grid.nodes), dtype=np.float64
            )
        return self.state.grid.nodes

    def price_slice(self, strikes: ArrayLike) -> DensityPricingResult:
        """Price calls and puts at arbitrary positive strikes."""
        strike_array = _validated_strikes(strikes)
        underlying = self.underlying_nodes
        weighted_density = self.state.grid.quadrature_weights * self.state.values
        calls = self.discount_factor * (
            np.maximum(underlying[:, None] - strike_array[None, :], 0.0).T
            @ weighted_density
        )
        puts = self.discount_factor * (
            np.maximum(strike_array[None, :] - underlying[:, None], 0.0).T
            @ weighted_density
        )
        mass = self.state.mass
        forward_moment = self.state.grid.integrate(underlying * self.state.values)
        parity_target = self.discount_factor * (
            forward_moment - strike_array * mass
        )
        residuals = calls - puts - parity_target
        return DensityPricingResult(
            strikes=strike_array,
            calls=calls,
            puts=puts,
            parity_residuals=residuals,
            mass=mass,
            forward_moment=forward_moment,
            discount_factor=self.discount_factor,
        )

    def price_vanillas(
        self, strikes: ArrayLike, option_types: ArrayLike
    ) -> FloatArray:
        """Price aligned call/put strikes using `C` and `P` option codes."""
        result = self.price_slice(strikes)
        types = np.asarray(option_types)
        if types.shape != result.strikes.shape:
            raise ValueError("option_types must have the same shape as strikes")
        if np.any((types != "C") & (types != "P")):
            raise ValueError("option_types entries must be 'C' or 'P'")
        return np.where(types == "C", result.calls, result.puts)

    def implied_volatilities(
        self,
        strikes: ArrayLike,
        option_types: ArrayLike,
        *,
        ttm: float,
    ) -> FloatArray:
        """Convert density prices to current SVM Black implied volatilities."""
        from vanilla_option_pricers import infer_bsm_implied_vol

        maturity = float(ttm)
        if not np.isfinite(maturity) or maturity <= 0.0:
            raise ValueError("ttm must be finite and positive")
        strike_array = _validated_strikes(strikes)
        types = np.asarray(option_types)
        prices = self.price_vanillas(strike_array, types)
        implied = np.empty(strike_array.size, dtype=np.float64)
        for index, (strike, option_type, price) in enumerate(
            zip(strike_array, types, prices)
        ):
            implied[index] = infer_bsm_implied_vol(
                forward=self.forward,
                ttm=maturity,
                strike=float(strike),
                given_price=float(price),
                discfactor=self.discount_factor,
                optiontype=str(option_type),
            )
        return implied


def _validated_strikes(values: ArrayLike) -> FloatArray:
    """Return a finite positive one-dimensional strike array."""
    strikes = np.array(values, dtype=np.float64, copy=True, order="C")
    if strikes.ndim != 1:
        raise ValueError("strikes must be one-dimensional")
    if not np.all(np.isfinite(strikes)) or np.any(strikes <= 0.0):
        raise ValueError("strikes must be finite and strictly positive")
    return strikes
