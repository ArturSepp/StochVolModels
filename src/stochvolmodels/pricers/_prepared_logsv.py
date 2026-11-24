"""Caller-owned LogSV expansion coefficients for vanilla pricing and volatility-state risk.

Prepare via LogSVPricer.prepare_fourier. Queries check resolution and tails in
premium or sensitivity units. Only explicit refinement solves additional frequencies.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
from numba.typed import List

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.logsv.affine_expansion import _state_basis
from stochvolmodels.models.logsv.model import _nonnegative_float, _positive_float
from stochvolmodels.utils import _fourier_quadrature as fq


def _freeze(values):
    array = np.asarray(values)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _query_default(value):
    """Only scalar tolerances transfer implicitly to different quotes."""
    if value is None or np.isscalar(value):
        return value
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return _ALIGNED


@dataclass(frozen=True, eq=False)
class _Quotes:
    expiry: int
    forward: float
    discount: float
    strikes: np.ndarray
    types: np.ndarray
    settings: Mapping
    sigma0: float
    sensitivity: bool = False

    def matches(self, expiry, forward, discount, strikes, types):
        return (
            self.expiry == expiry
            and self.forward == forward
            and self.discount == discount
            and np.array_equal(self.strikes, strikes)
            and np.array_equal(self.types, types)
        )

    def evaluate(self, coefficients: np.ndarray, theta: float) -> np.ndarray:
        """Return log M or dM/dsigma0 on the supplied frequencies."""
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            powers = _state_basis(self.sigma0 - theta, coefficients.shape[-1])
            log_mgf = coefficients @ powers
            if self.sensitivity:
                derivative = coefficients[:, 1:] @ (np.arange(1, powers.size) * powers[:-1])
                return np.exp(log_mgf) * derivative
        return log_mgf


@dataclass(frozen=True, eq=False, repr=False)
class PreparedLogSVGrid:
    """LogSV vanilla pricing handle for fixed dynamics, contour and prepared maturities.

    Use the pricer factory. Coefficients have shape (expiries, active nodes, terms).
    The quadrature diagnostics include the cumulative frequencies evaluated during
    preparation and subsequent refinement, including discarded parent panels.
    """

    _ttms: np.ndarray
    _transform: Callable
    _spot: bool
    _sigma0: float
    _theta: float
    _defaults: Mapping
    _limits: Mapping
    _quotes: tuple
    _plan: fq.FourierPlan
    _coefficients: np.ndarray
    diagnostics: fq.QuadratureDiagnostics

    @classmethod
    def _prepare(cls, chain, transform, spot, controls, sigma0, theta, n_terms):
        sigma0 = _nonnegative_float(sigma0, "sigma0")
        ttms = _freeze(np.asarray(chain.ttms, dtype=float))
        if not ttms.size or ttms.ndim != 1 or np.any(np.diff(ttms) <= 0):
            raise ValueError("preparation requires increasing maturities")
        settings = fq.vanilla_settings_by_expiry(controls, chain.strikes_ttms)
        quotes = tuple(
            _Quotes(
                i,
                float(f),
                float(d),
                _freeze(np.asarray(k, dtype=float)),
                _freeze(np.asarray(types, dtype=str)),
                _row_settings(s),
                sigma0,
            )
            for i, (f, d, k, types, s) in enumerate(
                zip(
                    chain.forwards,
                    chain.discfactors,
                    chain.strikes_ttms,
                    chain.optiontypes_ttms,
                    settings,
                    strict=True,
                )
            )
        )
        if len(quotes) != ttms.size:
            raise ValueError("supply one quote row per expiry")
        limits = {
            k: v.item() if isinstance(v, np.ndarray) and v.ndim == 0 else v
            for k, v in controls.items() if k not in _PRICE_KEYS
        }
        defaults = {k: _query_default(v) for k, v in controls.items() if k in _PRICE_KEYS}
        plan, coefficients, diagnostics = _build(ttms, transform, spot, quotes, limits, theta, n_terms)
        return cls(
            ttms,
            transform,
            spot,
            sigma0,
            theta,
            MappingProxyType(defaults),
            MappingProxyType(limits),
            quotes,
            plan,
            coefficients,
            diagnostics,
        )

    def _requests(self, chain, fourier, sigma0, *, sensitivity=False):
        state = self._sigma0 if sigma0 is None else _nonnegative_float(sigma0, "sigma0")
        overrides = {} if fourier is None else dict(fourier)
        if overrides.keys() - _PRICE_KEYS:
            raise ValueError("query settings accept price_atol, price_rtol and time_values")
        rows = fq.vanilla_settings_by_expiry(overrides, chain.strikes_ttms)
        result = []
        for t, f, d, k, types, row in zip(
            chain.ttms,
            chain.forwards,
            chain.discfactors,
            chain.strikes_ttms,
            chain.optiontypes_ttms,
            rows,
            strict=True,
        ):
            i = int(np.searchsorted(self._ttms, t))
            if i == self._ttms.size or self._ttms[i] != t:
                raise ValueError(f"maturity {t:g} was not prepared")
            settings = {} if sensitivity else dict(self._defaults)
            settings.update(row)
            for key, value in settings.items():
                if value is _ALIGNED:
                    if key == "time_values" and not settings.get("price_rtol", 0):
                        settings[key] = None
                        continue
                    original = next((q for q in self._quotes
                                     if not q.sensitivity and q.matches(i, f, d, k, types)), None)
                    if original is None or key not in original.settings:
                        raise ValueError(f"supply strike-aligned {key} for the new quotes")
                    settings[key] = original.settings[key]
            result.append(
                _Quotes(
                    i,
                    float(f),
                    float(d),
                    _freeze(np.asarray(k, dtype=float)),
                    _freeze(np.asarray(types, dtype=str)),
                    _row_settings(settings),
                    state,
                    sensitivity,
                )
            )
        if not result:
            raise ValueError("supply at least one prepared maturity")
        return tuple(result)

    def price_chain(self, quotes: OptionChain, *, sigma0: float | None = None,
                    fourier: Mapping | None = None) -> List:
        """Price new quotes at prepared maturities with zero additional coefficient solves.

        A failed premium-resolution or tail check raises a diagnostic FloatingPointError.
        Pass quote-specific price_atol, price_rtol and time_values in the optional mapping.
        sigma0 is the initial annualized volatility in decimal units. Its default is
        the preparation state. All other parameters and calendar loadings stay fixed.
        """
        return self._query(self._requests(quotes, fourier, sigma0))

    def sigma0_sensitivity(self, quotes: OptionChain, *, sigma0: float | None = None,
                           atol: float = 1e-6) -> List:
        """Return d(discounted premium)/d(sigma0) at fixed forwards and model dynamics.

        Multiply by 0.01 for the first-order effect of one volatility point. The
        differentiated integral has its own absolute tolerance per unit sigma0.
        A failed check raises with no coefficient solves.
        """
        settings = {"price_atol": _positive_float(atol, "atol")}
        return self._query(self._requests(quotes, settings, sigma0, sensitivity=True))

    def _query(self, rows: tuple) -> List:
        def saved(phi):
            return (row.evaluate(self._coefficients[row.expiry], self._theta) for row in rows)

        _, prices, _ = _integrate(
            self._ttms,
            self._spot,
            rows,
            self._limits,
            saved,
            fixed_plan=self._plan,
            initial_refinements=self.diagnostics.refinements,
            discarded_evaluations=self.diagnostics.evaluations - self._plan.nodes.size,
        )
        return List(prices)

    def refined_for(self, quotes: OptionChain, *, sigma0: float | None = None,
                    fourier: Mapping | None = None, sensitivity_atol: float | None = None):
        """Return a handle checked for original and new quotes within the preparation limits.

        Only new frequencies incur coefficient solves. The original remains usable on failure.
        Supply sensitivity_atol to certify state sensitivities as well as premiums.
        """
        additional = []
        requests = self._requests(quotes, fourier, sigma0)
        if sensitivity_atol is not None:
            settings = {"price_atol": _positive_float(sensitivity_atol, "sensitivity_atol")}
            requests += self._requests(quotes, settings, sigma0, sensitivity=True)
        for new in requests:
            budget = fq._premium_budget(new.strikes, **new.settings)
            matching = (
                q
                for q in self._quotes
                if q.sigma0 == new.sigma0 and q.sensitivity == new.sensitivity
                and q.matches(new.expiry, new.forward, new.discount, new.strikes, new.types)
            )
            if not any(
                np.all(fq._premium_budget(q.strikes, **q.settings) <= budget)
                for q in matching
            ):
                additional.append(new)
        if not additional:
            return self
        rows = self._quotes + tuple(additional)
        plan, coefficients, diagnostics = _build(
            self._ttms, self._transform, self._spot, rows, self._limits, self._theta,
            self._coefficients.shape[-1], previous=self,
        )
        return replace(self, _quotes=rows, _plan=plan, _coefficients=coefficients, diagnostics=diagnostics)


_PRICE_KEYS = {"price_atol", "price_rtol", "time_values"}
_ALIGNED = object()


def _row_settings(settings):
    return MappingProxyType(
        {
            k: _freeze(v) if _query_default(v) is _ALIGNED else _query_default(v)
            for k, v in settings.items()
            if k in _PRICE_KEYS
        }
    )


def _integrate(ttms, spot, rows, limits, sampler, **plan_options):
    return fq.integrate_vanilla_chain(
        sampler,
        ttms=[ttms[q.expiry] for q in rows],
        forwards=[q.forward for q in rows],
        discfactors=[q.discount for q in rows],
        strikes_ttms=[q.strikes for q in rows],
        optiontypes_ttms=[q.types for q in rows],
        settings=[{**limits, **q.settings} for q in rows],
        is_spot_measure=spot,
        grouped_quotes=True,
        derivative_rows=[q.sensitivity for q in rows],
        **plan_options,
    )


def _build(ttms, transform, spot, rows, limits, theta, n_terms, previous=None):
    # Transient lookup also covers newly sampled panels later discarded by bisection.
    columns = {}
    options = {}
    if previous is not None:
        columns.update(zip(previous._plan.nodes, previous._coefficients.swapaxes(0, 1), strict=True))
        options.update(
            initial_edges=previous._plan.edges,
            initial_refinements=previous.diagnostics.refinements,
            discarded_evaluations=previous.diagnostics.evaluations - len(columns),
        )

    def sample(phi):
        missing = np.array([z.imag not in columns for z in phi])
        known = ~missing
        matrix = np.empty((len(ttms), phi.size, n_terms), dtype=complex)
        if known.any():
            matrix[:, known] = np.stack([columns[u] for u in phi.imag[known]], axis=1)
        computed = 0
        if missing.any():
            new_values = iter(transform(phi[missing]))
        else:
            computed = len(ttms)
        # Seed quotes cover every expiry in schedule order before extension quotes.
        # Yield each transform as it is computed so failures retain their expiry.
        for q in rows:
            while computed <= q.expiry:
                values = np.asarray(next(new_values, None))
                if values.shape != (np.count_nonzero(missing), n_terms):
                    raise ValueError("supply one expansion coefficient vector per requested frequency")
                matrix[computed, missing] = values
                computed += 1
            yield q.evaluate(matrix[q.expiry], theta)
        columns.update(zip(phi.imag[missing], matrix[:, missing].swapaxes(0, 1), strict=True))

    plan, _, diagnostics = _integrate(ttms, spot, rows, limits, sample, **options)
    if previous is not None and np.array_equal(plan.edges, previous._plan.edges):
        return previous._plan, previous._coefficients, diagnostics
    coefficients = _freeze(np.stack([columns[u] for u in plan.nodes], axis=1))
    return plan, coefficients, diagnostics
