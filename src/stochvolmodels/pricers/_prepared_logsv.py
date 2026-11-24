"""Caller-owned LogSV expansion coefficients for vanilla pricing and forward and volatility risk.

Prepare via LogSVPricer.prepare_fourier. Queries check resolution and tails in
premium or sensitivity units. Only explicit refinement solves additional frequencies.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

import numpy as np
from numba.typed import List

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.logsv.affine_expansion import _state_basis
from stochvolmodels.models.logsv.model import _nonnegative_float
from stochvolmodels.pricers._prepared_fourier import _PreparedFourierGrid, _VanillaQuotes


@dataclass(frozen=True, eq=False)
class _Quotes(_VanillaQuotes):
    sigma0: float
    theta: float

    def same_output(self, other):
        """Match the requested derivative and initial volatility for this expansion."""
        return self.sigma0 == other.sigma0 and self.quantity == other.quantity

    def evaluate(self, coefficients: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        """Return log M for prices or the numerator of the differentiated half-contour integral."""
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            powers = _state_basis(self.sigma0 - self.theta, coefficients.shape[-1])
            log_mgf = coefficients @ powers
            if self.quantity == "state":
                derivative = coefficients[:, 1:] @ (np.arange(1, powers.size) * powers[:-1])
                return np.exp(log_mgf) * derivative
        return super().evaluate(log_mgf, nodes)


@dataclass(frozen=True, eq=False, repr=False)
class PreparedLogSVGrid(_PreparedFourierGrid):
    """LogSV vanilla pricing handle for fixed dynamics, contour and prepared maturities.

    Use the pricer factory. Coefficients have shape (expiries, active nodes, terms).
    The quadrature diagnostics include the cumulative frequencies evaluated during
    preparation and subsequent refinement, including discarded parent panels.
    """

    _sigma0: float
    _theta: float

    @property
    def _coefficients(self):
        """Expansion coefficient vectors in maturity and frequency order."""
        return self._samples

    @classmethod
    def _prepare(cls, chain, transform, spot, controls, sigma0, theta, n_terms):
        sigma0 = _nonnegative_float(sigma0, "sigma0")
        return cls._prepare_grid(
            chain, transform, spot, controls, (n_terms,),
            row_factory=partial(_Quotes, sigma0=sigma0, theta=theta), _sigma0=sigma0, _theta=theta,
        )

    def _row_factory(self, sigma0):
        state = self._sigma0 if sigma0 is None else _nonnegative_float(sigma0, "sigma0")
        return partial(_Quotes, theta=self._theta, sigma0=state)

    def price_chain(self, quotes: OptionChain, *, sigma0: float | None = None,
                    fourier: Mapping | None = None) -> List:
        """Price new quotes at prepared maturities with zero additional coefficient solves.

        A failed premium-resolution or tail check raises a diagnostic FloatingPointError.
        Pass quote-specific price_atol, price_rtol and time_values in the optional mapping.
        sigma0 is the initial annualized volatility in decimal units. Its default is
        the preparation state. All other parameters and calendar loadings stay fixed.
        """
        rows = self._quote_requests(quotes, fourier, row_factory=self._row_factory(sigma0))
        return List(self._query(rows))

    def sigma0_sensitivity(self, quotes: OptionChain, *, sigma0: float | None = None,
                           atol: float = 1e-6) -> List:
        """Return d(discounted premium)/d(sigma0) at fixed forwards and model dynamics.

        Multiply by 0.01 for the first-order effect of one volatility point. The
        differentiated integral has its own absolute tolerance per unit sigma0.
        A failed check raises with no coefficient solves.
        """
        return List(self._query(self._risk_requests(quotes, sigma0, "state", atol, "atol")))

    def _risk_requests(self, quotes, sigma0, quantity, atol, name):
        return self._sensitivity_requests(quotes, quantity, atol, name,
                                          row_factory=self._row_factory(sigma0))

    def forward_greeks(self, quotes: OptionChain, *, sigma0: float | None = None,
                       delta_atol: float = 1e-6, gamma_atol: float = 1e-6) -> tuple[List, List]:
        """Return (forward deltas, forward gammas) of the discounted vanilla premiums.

        Each list contains one array per requested expiry, in input strike order.
        Strike, discount, initial volatility and model dynamics stay fixed as the
        forward varies. On the inverse contour the premium includes the pricer's
        forward conversion. Absolute tolerances are in premium per forward unit
        and premium per squared forward unit, respectively. Queries use saved
        coefficients. If either integral fails its checks, call refined_for with
        the same state and tolerances.
        """
        deltas = self._risk_requests(quotes, sigma0, "delta", delta_atol, "delta_atol")
        gammas = self._risk_requests(quotes, sigma0, "gamma", gamma_atol, "gamma_atol")
        values = self._query(deltas + gammas)
        return List(values[:len(deltas)]), List(values[len(deltas):])

    def refined_for(self, quotes: OptionChain, *, sigma0: float | None = None,
                    fourier: Mapping | None = None, sensitivity_atol: float | None = None,
                    delta_atol: float | None = None, gamma_atol: float | None = None):
        """Return a handle checked for original and new quotes within the preparation limits.

        Only new frequencies incur coefficient solves. The original remains usable on failure.
        Supply sensitivity_atol for volatility-state sensitivities, or delta_atol
        and gamma_atol to check the respective forward derivatives as well as premiums.
        """
        requests = self._quote_requests(quotes, fourier, row_factory=self._row_factory(sigma0))
        for quantity, atol, name in (("state", sensitivity_atol, "sensitivity_atol"),
                                     ("delta", delta_atol, "delta_atol"),
                                     ("gamma", gamma_atol, "gamma_atol")):
            if atol is not None:
                requests += self._risk_requests(quotes, sigma0, quantity, atol, name)
        return self._refine(requests)
