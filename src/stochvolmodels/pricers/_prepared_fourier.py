"""Caller-owned Fourier samples and quote-specific error checks for European vanillas.

Prepared grids bind the transform, measure and maturities. Explicit refinement can
add frequencies while keeping accepted samples and cumulative quadrature work.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from numbers import Real
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from stochvolmodels.utils import _fourier_quadrature as fq


def _freeze(values):
    """Take a read-only copy of the values and preserve their shape."""
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
class _VanillaQuotes:
    """Snapshot quotes and their tolerance for one maturity and output."""
    expiry: int
    forward: float
    discount: float
    strikes: np.ndarray
    types: np.ndarray
    settings: Mapping
    quantity: fq._Quantity = field(default="price", kw_only=True)

    def matches(self, expiry, forward, discount, strikes, types):
        """Check whether the same maturity, payoff inputs and strike order were used."""
        return (
            self.expiry == expiry
            and self.forward == forward
            and self.discount == discount
            and np.array_equal(self.strikes, strikes)
            and np.array_equal(self.types, types)
        )

    def evaluate(self, samples, nodes):
        """Return log M for premiums or the differentiated forward numerator."""
        # The strike phase is proportional to F**(1/2-iu). The quadrature applies
        # the remaining 1/F or 1/F**2 in premium units after integration.
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            if self.quantity == "delta":
                return np.exp(samples) * (.5 - 1j * nodes)
            if self.quantity == "gamma":
                return -np.exp(samples) * (nodes**2 + .25)
        return samples

    def same_output(self, other):
        """Whether both rows request the same quantity at the same model state."""
        return self.quantity == other.quantity

    def covered_by(self, rows):
        """Whether an identical output has already passed an equal or tighter budget."""
        budget = fq._premium_budget(self.strikes, **self.settings)
        return any(
            row.same_output(self)
            and row.matches(self.expiry, self.forward, self.discount, self.strikes, self.types)
            and np.all(fq._premium_budget(row.strikes, **row.settings) <= budget)
            for row in rows
        )


@dataclass(frozen=True, eq=False, repr=False)
class _PreparedFourierGrid:
    """Shared panel geometry and immutable values indexed by maturity and frequency."""

    _ttms: np.ndarray
    _transform: Callable
    _spot: bool
    _defaults: Mapping
    _limits: Mapping
    _quotes: tuple
    _plan: fq.FourierPlan
    _samples: np.ndarray
    diagnostics: fq.QuadratureDiagnostics
    _seed_panels: ClassVar[int] = 9

    @classmethod
    def _prepare_grid(cls, chain, transform, spot, controls, sample_shape, *,
                      row_factory=_VanillaQuotes, **model_fields):
        """Check the seed quotes and construct a reusable handle with frozen samples."""
        ttms = _freeze(np.asarray(chain.ttms, dtype=float))
        if not ttms.size or ttms.ndim != 1 or np.any(np.diff(ttms) <= 0):
            raise ValueError("preparation requires increasing maturities")
        settings = fq.vanilla_settings_by_expiry(controls, chain.strikes_ttms)
        quotes = tuple(
            row_factory(
                i,
                float(f),
                float(d),
                _freeze(np.asarray(k, dtype=float)),
                _freeze(np.asarray(types, dtype=str)),
                _row_settings(s),
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
        plan, samples, diagnostics = _build(ttms, transform, spot, quotes, limits, sample_shape,
                                            seed_panels=cls._seed_panels)
        return cls(
            ttms,
            transform,
            spot,
            MappingProxyType(defaults),
            MappingProxyType(limits),
            quotes,
            plan,
            samples,
            diagnostics,
            **model_fields,
        )

    def _quote_requests(self, chain, fourier, *, row_factory=_VanillaQuotes, use_defaults=True):
        """Align the query quotes to the saved maturity indices and tolerance settings."""
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
            settings = dict(self._defaults) if use_defaults else {}
            settings.update(row)
            for key, value in settings.items():
                if value is _ALIGNED:
                    if key == "time_values" and not settings.get("price_rtol", 0):
                        settings[key] = None
                        continue
                    original = next((q for q in self._quotes
                                     if q.quantity == "price" and q.matches(i, f, d, k, types)), None)
                    if original is None or key not in original.settings:
                        raise ValueError(f"supply strike-aligned {key} for the new quotes")
                    settings[key] = original.settings[key]
            result.append(
                row_factory(
                    i,
                    float(f),
                    float(d),
                    _freeze(np.asarray(k, dtype=float)),
                    _freeze(np.asarray(types, dtype=str)),
                    _row_settings(settings),
                )
            )
        if not result:
            raise ValueError("supply at least one prepared maturity")
        return tuple(result)

    def _sensitivity_requests(self, quotes, quantity, atol, name, *, row_factory=_VanillaQuotes):
        """Build derivative rows with their own positive scalar tolerance."""
        if (isinstance(atol, (bool, np.bool_)) or not isinstance(atol, Real)
                or not np.isfinite(float(atol)) or atol <= 0):
            raise ValueError(f"{name} must be a strictly positive finite scalar")
        return self._quote_requests(
            quotes, {"price_atol": float(atol)}, use_defaults=False,
            row_factory=partial(row_factory, quantity=quantity),
        )

    def _query(self, rows: tuple) -> list[np.ndarray]:
        """Validate and value the rows on the stored plan without sampling new nodes."""
        def saved(phi):
            """Return the saved transform or derivative for each requested row."""
            return (row.evaluate(self._samples[row.expiry], phi.imag) for row in rows)

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
        return prices

    def _refine(self, requests):
        """Keep existing certificates and refine for uncovered requests."""
        additional = tuple(row for row in requests if not row.covered_by(self._quotes))
        if not additional:
            return self
        rows = self._quotes + additional
        plan, samples, diagnostics = _build(
            self._ttms, self._transform, self._spot, rows, self._limits, self._samples.shape[2:],
            seed_panels=self._seed_panels, previous=self,
        )
        return replace(self, _quotes=rows, _plan=plan, _samples=samples, diagnostics=diagnostics)


_PRICE_KEYS = {"price_atol", "price_rtol", "time_values"}
_ALIGNED = object()


def _row_settings(settings):
    """Snapshot the scalar and strike-aligned tolerance inputs."""
    return MappingProxyType(
        {
            k: _freeze(v) if _query_default(v) is _ALIGNED else _query_default(v)
            for k, v in settings.items()
            if k in _PRICE_KEYS
        }
    )


def _integrate(ttms, spot, rows, limits, sampler, **plan_options):
    """Send prepared quote rows and their tolerances to the chain quadrature."""
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
        quantities=[q.quantity for q in rows],
        **plan_options,
    )


def _build(ttms, transform, spot, rows, limits, sample_shape, *, seed_panels, previous=None):
    """Build an accepted plan and retain its transform samples by maturity."""
    # Transient lookup also covers newly sampled panels later discarded by bisection.
    columns = {}
    options = dict(seed_panels=seed_panels)
    if previous is not None:
        columns.update(zip(previous._plan.nodes, previous._samples.swapaxes(0, 1), strict=True))
        options.update(
            initial_edges=previous._plan.edges,
            initial_refinements=previous.diagnostics.refinements,
            discarded_evaluations=previous.diagnostics.evaluations - len(columns),
        )

    def sample(phi):
        """Sample missing frequencies once and yield quote rows in order."""
        missing = np.array([z.imag not in columns for z in phi])
        known = ~missing
        matrix = np.empty((len(ttms), phi.size, *sample_shape), dtype=complex)
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
                if values.shape != (np.count_nonzero(missing), *sample_shape):
                    raise ValueError("unexpected transform sample shape for the requested frequencies")
                matrix[computed, missing] = values
                computed += 1
            yield q.evaluate(matrix[q.expiry], phi.imag)
        columns.update(zip(phi.imag[missing], matrix[:, missing].swapaxes(0, 1), strict=True))

    plan, _, diagnostics = _integrate(ttms, spot, rows, limits, sample, **options)
    if previous is not None and np.array_equal(plan.edges, previous._plan.edges):
        return previous._plan, previous._samples, diagnostics
    samples = _freeze(np.stack([columns[u] for u in plan.nodes], axis=1))
    return plan, samples, diagnostics
