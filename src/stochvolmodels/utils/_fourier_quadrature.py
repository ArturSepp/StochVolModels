"""Embedded Gauss–Kronrod integration for European vanilla slices and chains.

Frequencies are the imaginary part of the log-return contour. A plan contains
only geometry and weights for du. Resolution and empirical tail estimates use
discounted premium units for each strike. Model samples belong to a single
invocation and can be discarded once the final plan has priced the slice.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from stochvolmodels.utils.mgf_pricer import _vanilla_slice_with_weights


# QUADPACK DQK15, Piessens et al. (1983), https://netlib.org/quadpack/dqk15.f
# Nonnegative nodes in ascending order; columns are Kronrod and embedded Gauss weights.
_RULE = np.array([
    [0.0,                0.20948214108472783, 0.41795918367346939],
    [0.20778495500789847, 0.20443294007529889, 0.0],
    [0.40584515137739717, 0.19035057806478541, 0.38183005050511894],
    [0.58608723546769113, 0.16900472663926790, 0.0],
    [0.74153118559939444, 0.14065325971552592, 0.27970539148927667],
    [0.86486442335976907, 0.10479001032225018, 0.0],
    [0.94910791234275852, 0.06309209262997855, 0.12948496616886969],
    [0.99145537112081264, 0.02293532201052922, 0.0],
])
_POSITIVE_NODES, _KRONROD, _GAUSS = _RULE.T
_NODES = np.r_[-_POSITIVE_NODES[:0:-1], _POSITIVE_NODES]
_FINE = np.r_[_KRONROD[:0:-1], _KRONROD]
_EMBEDDED = np.r_[_GAUSS[:0:-1], _GAUSS]

_Quantity = Literal["price", "state", "delta", "gamma"]


def _half_contour_scale(forward, strikes, discount, quantity):
    """Discounted sqrt(F*K)/pi scale including any forward differentiation."""
    if quantity in ("delta", "gamma"):
        scale = discount * (np.sqrt(strikes) / np.sqrt(forward)) / np.pi
        return scale / forward if quantity == "gamma" else scale
    return discount * np.sqrt(forward) * np.sqrt(strikes) / np.pi


@dataclass(frozen=True, eq=False)
class FourierPlan:
    """Read-only consecutive panels on [0, U], with flattened nodes and aligned du weights."""

    edges: np.ndarray
    nodes: np.ndarray = field(init=False, repr=False)
    weights: np.ndarray = field(init=False, repr=False)
    embedded_weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        edges = np.array(self.edges, dtype=float, copy=True)
        if (edges.ndim != 1 or edges.size < 2 or edges[0] != 0.0
                or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0)):
            raise ValueError("panel edges must increase strictly from zero to a finite cutoff")
        nodes, weights, embedded = _map_panels(edges)
        for name, values in (("edges", edges), ("nodes", nodes.ravel()),
                             ("weights", weights.ravel()), ("embedded_weights", embedded.ravel())):
            object.__setattr__(self, name, np.frombuffer(values.tobytes(), dtype=float))

    def price_vanillas(
        self, log_mgf: np.ndarray, *, forward: float, strikes: np.ndarray, optiontypes: np.ndarray,
        discfactor: float = 1.0, is_spot_measure: bool = True,
    ) -> np.ndarray:
        """Apply frozen weights to new samples. Accuracy for new parameters needs rechecking."""
        strikes, optiontypes = _checked_quotes(
            forward, discfactor, strikes, optiontypes, is_spot_measure,
        )
        values = np.asarray(log_mgf, dtype=complex)
        if values.shape != self.nodes.shape:
            raise ValueError("the fixed grid requires one log transform per frequency")
        if not np.isfinite(values).all():
            raise FloatingPointError("nonfinite log transform on fixed grid")
        phi = (-0.5 if is_spot_measure else 0.5) + 1j * self.nodes
        prices = _vanilla_slice_with_weights(
            values, phi, self.weights, forward, strikes, optiontypes, discfactor, is_spot_measure,
        )
        if not np.isfinite(prices).all():
            raise FloatingPointError("nonfinite vanilla premium")
        return prices

    def vanilla_sensitivities(self, derivatives: np.ndarray, *, forward: float,
                              strikes: np.ndarray, discfactor: float,
                              quantity: _Quantity | tuple[_Quantity, ...] = "state") -> np.ndarray:
        """Invert sensitivity columns with the same strike phases and column-specific units."""
        if derivatives.ndim not in (1, 2) or derivatives.shape[0] != self.nodes.size:
            raise ValueError("supply one transform derivative row per Fourier node")
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            flat = derivatives.reshape(self.nodes.size, -1)
            weighted = flat * (self.weights / (self.nodes**2 + .25))[:, None]
            integrals = np.array([np.exp(-1j*x*self.nodes) @ weighted
                                  for x in np.log(forward) - np.log(strikes)])
            if isinstance(quantity, tuple):
                if len(quantity) != flat.shape[1]:
                    raise ValueError("supply one sensitivity quantity per column")
                scales = np.column_stack([_half_contour_scale(forward, strikes, discfactor, q)
                                          for q in quantity])
            else:
                scales = _half_contour_scale(forward, strikes, discfactor, quantity)[:, None]
            values = -scales * integrals.real
        if not np.isfinite(values).all():
            raise FloatingPointError("nonfinite vanilla sensitivity")
        return values[:, 0] if derivatives.ndim == 1 else values


def _checked_quotes(
    forward: float, discfactor: float, strikes: np.ndarray, optiontypes: np.ndarray,
    is_spot_measure: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the slice and normalize contract codes for the compiled payoff function."""
    if not np.isfinite(forward) or forward <= 0 or not np.isfinite(discfactor) or discfactor <= 0:
        raise ValueError("forward and discount must be positive and finite")
    strikes = np.asarray(strikes, dtype=float)
    optiontypes = np.asarray(optiontypes, dtype=str)
    if (strikes.ndim != 1 or not strikes.size or not np.isfinite(strikes).all()
            or np.any(strikes <= 0) or optiontypes.shape != strikes.shape):
        raise ValueError("supply a positive finite strike and an option type for each quote")
    allowed = ("C", "P") if is_spot_measure else ("IC", "IP", "C", "P")
    if not np.isin(optiontypes, allowed).all():
        raise ValueError("unsupported option type for the requested measure")
    return strikes, optiontypes


def _map_panels(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map the nested pair to consecutive intervals, including intervals starting above zero."""
    half_width = 0.5 * np.diff(edges)[:, None]
    midpoint = edges[:-1, None] + half_width
    nodes = midpoint + half_width * _NODES
    if (np.any(nodes[:, 0] <= edges[:-1]) or np.any(nodes[:, -1] >= edges[1:])
            or np.any(np.diff(nodes, axis=1) <= 0)):
        raise FloatingPointError("panel nodes cannot be resolved in floating point")
    return nodes, half_width * _FINE, half_width * _EMBEDDED


@dataclass(frozen=True)
class QuadratureDiagnostics:
    """Work performed and estimated errors at the quote nearest to or above its tolerance."""

    maturity: float
    strike: float
    cutoff: float
    panels: int
    evaluations: int
    refinements: int
    resolution_error: float
    tail_estimate: float


class QuadratureError(FloatingPointError):
    """A failed integral with diagnostics for the last evaluated partition."""

    def __init__(self, reason: str, diagnostics: QuadratureDiagnostics):
        self.diagnostics = diagnostics
        super().__init__(f"{reason}. {diagnostics}")


@dataclass
class _Panel:
    right: float
    shell: int
    samples: np.ndarray
    error: np.ndarray
    mass: np.ndarray


def uses_controlled_fourier(fourier: object) -> bool:
    """Select the public opt-in and reject unrecognized mode names."""
    if fourier is None or (isinstance(fourier, str) and fourier == "legacy"):
        return False
    if isinstance(fourier, Mapping):
        return True
    raise ValueError("fourier must be a settings mapping, 'legacy' or None")


def _quote_rows(value, strikes_ttms: Sequence[np.ndarray], name: str) -> list[np.ndarray]:
    """Align one premium setting with each quote, preserving the chain's row order."""
    if np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0):
        return [np.full(len(strikes), value, dtype=float) for strikes in strikes_ttms]
    if len(strikes_ttms) == 1 and np.asarray(value).ndim == 1:
        value = [value]
    if value is not None and len(value) == len(strikes_ttms):
        rows = [np.asarray(row, dtype=float) for row in value]
        if all(row.shape == np.shape(strikes) for row, strikes in zip(rows, strikes_ttms)):
            return rows
    raise ValueError(f"{name} requires a scalar or one strike-aligned vector per expiry")


def vanilla_settings_by_expiry(
    fourier: Mapping[str, object], strikes_ttms: Sequence[np.ndarray],
) -> list[dict]:
    """Separate shared work limits from strike-aligned tolerances and time values."""
    allowed = {"price_atol", "price_rtol", "time_values", "initial_cutoff", "max_cutoff",
               "max_nodes", "max_refinements"}
    unknown = fourier.keys() - allowed
    if unknown:
        raise ValueError(f"unknown fourier settings: {', '.join(sorted(map(str, unknown)))}")
    settings = [dict(fourier) for _ in strikes_ttms]
    for name in ("price_atol", "time_values"):
        if name in fourier and not (name == "time_values" and fourier[name] is None):
            for row, value in zip(settings, _quote_rows(fourier[name], strikes_ttms, name)):
                row[name] = value
    return settings


def integrate_vanilla_slice(
    log_mgf: Callable[[np.ndarray], np.ndarray], *, maturity: float, forward: float,
    strikes: np.ndarray, optiontypes: np.ndarray, discfactor: float = 1.0,
    is_spot_measure: bool = True, price_atol: float | np.ndarray = 1e-6,
    price_rtol: float = 0.0, time_values: np.ndarray | None = None,
    initial_cutoff: float = 1.0, max_cutoff: float = 16384.0,
    max_nodes: int = 32767, max_refinements: int = 512,
) -> tuple[FourierPlan, np.ndarray, QuadratureDiagnostics]:
    """Build a plan and price all strikes from one set of model samples.

    The callback takes a batch of complex contour points and returns the corresponding
    log transforms. Both half contours give the premium scale D sqrt(F K): under the
    spot measure the multiplier is D K exp(log(F/K)/2), under the inverse measure it
    is D F exp(-log(F/K)/2).

    Each quote gets half its tolerance for summed absolute embedded differences and
    half for each of two consecutive outer frequency shells. Subdividing a shell
    preserves its total mass. The latter check is empirical. A relative tolerance
    uses caller-supplied, fixed time values in premium units. The evaluation limit
    counts every sampled frequency, including samples in subsequently bisected panels.
    """
    settings = dict(price_atol=price_atol, price_rtol=price_rtol, time_values=time_values,
                    initial_cutoff=initial_cutoff, max_cutoff=max_cutoff,
                    max_nodes=max_nodes, max_refinements=max_refinements)
    plan, prices, diagnostics = integrate_vanilla_chain(
        lambda phi: (log_mgf(phi),), ttms=[maturity], forwards=[forward],
        strikes_ttms=[strikes], optiontypes_ttms=[optiontypes], discfactors=[discfactor],
        is_spot_measure=is_spot_measure, settings=[settings], seed_panels=1,
    )
    return plan, prices[0], diagnostics


def _premium_budget(strikes, price_atol=1e-6, price_rtol=0.0, time_values=None):
    """Allocate half of each discounted premium tolerance to either error check."""
    tolerance = np.broadcast_to(np.asarray(price_atol, dtype=float), strikes.shape)
    if not np.isfinite(tolerance).all() or np.any(tolerance <= 0):
        raise ValueError("absolute premium tolerances must be positive and finite")
    if not np.isfinite(price_rtol) or price_rtol < 0:
        raise ValueError("relative tolerance must be finite and nonnegative")
    if price_rtol:
        scales = np.asarray(time_values, dtype=float)
        if scales.shape != strikes.shape or not np.isfinite(scales).all() or np.any(scales < 0):
            raise ValueError("relative tolerance needs a finite nonnegative time value per strike")
        tolerance = tolerance + price_rtol * scales
    budget = 0.5 * tolerance
    if not np.isfinite(budget).all() or np.any(budget <= 0):
        raise ValueError("combined premium tolerance is outside the numerical range")
    return budget


def integrate_vanilla_chain(
    log_mgfs: Callable[[np.ndarray], Iterable[np.ndarray]], *, ttms, forwards, strikes_ttms,
    optiontypes_ttms, discfactors, is_spot_measure: bool, settings: Sequence[dict],
    seed_panels: int = 9, scenario_count: int = 1, initial_edges: np.ndarray | None = None,
    initial_refinements: int = 0, discarded_evaluations: int = 0,
    fixed_plan: FourierPlan | None = None, grouped_quotes: bool = False,
    quantities: Sequence[_Quantity] | None = None,
) -> tuple[FourierPlan, list[np.ndarray], QuadratureDiagnostics]:
    """Preflight all quotes together and value every expiry on the accepted frozen grid.

    Each transform batch has one row per expiry and one column per new frequency.
    All strikes at a maturity share that row. The initial batch covers consecutive
    dyadic shells. Later bisections and extensions request only new nodes. Active
    panel samples are retained in grid order for valuation with no additional solves.
    The node limit counts distinct frequencies across the chain. Multiple parameter
    scenarios contribute to the same panel errors and tail checks. Their transform
    rows are yielded in scenario-major order. A validation run can start from a
    prescribed refinement of a previous partition, retaining the original dyadic shells.
    A derivative row supplies the differentiated numerator of the half-contour
    integral, with inverse forward powers factored into its quote scale. Its embedded
    and tail checks use the derivative's units and tolerances. Forward delta includes
    the derivative of the additive call term.
    """
    rows = [_checked_quotes(f, d, k, types, is_spot_measure)
            for f, d, k, types in zip(forwards, discfactors, strikes_ttms, optiontypes_ttms,
                                      strict=True)]
    ttms = np.asarray(ttms, dtype=float)
    if (ttms.shape != (len(rows),) or not rows or len(settings) != len(rows)
            or not np.isfinite(ttms).all() or np.any(ttms <= 0)
            or (not grouped_quotes and np.any(np.diff(ttms) <= 0))):
        raise ValueError("require increasing finite positive maturities and one row per expiry")
    common = settings[0]
    initial_cutoff = common.get("initial_cutoff", 0.5)
    max_cutoff = common.get("max_cutoff", 16384.0)
    max_nodes = common.get("max_nodes", 32767)
    max_refinements = common.get("max_refinements", 512)
    if (not np.isfinite([initial_cutoff, max_cutoff]).all()
            or initial_cutoff <= 0 or max_cutoff < initial_cutoff):
        raise ValueError("require ordered positive finite cutoffs")
    for value in (max_nodes, max_refinements):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("work limits must be nonnegative integers")
    quantities = ["price"] * len(rows) if quantities is None else list(quantities)
    if len(quantities) != len(rows):
        raise ValueError("supply one output quantity per quote row")
    if scenario_count > 1:
        rows = rows * scenario_count
        quantities = quantities * scenario_count
        settings = list(settings) * scenario_count
        ttms = np.tile(ttms, scenario_count)
        forwards = np.tile(forwards, scenario_count)
        discfactors = np.tile(discfactors, scenario_count)
    budget = np.concatenate([
        _premium_budget(k, s.get("price_atol", 1e-6), s.get("price_rtol", 0), s.get("time_values"))
        for (k, _), s in zip(rows, settings)
    ])
    strikes = np.concatenate([k for k, _ in rows])
    expiry = np.repeat(np.arange(len(rows)), [len(k) for k, _ in rows])
    forward = np.asarray(forwards)[expiry]

    real_phi = -0.5 if is_spot_measure else 0.5
    log_moneyness = np.log(forward) - np.log(strikes)
    premium_scale = np.concatenate([_half_contour_scale(f, k, d, quantity)
                                    for f, (k, _), d, quantity in
                                    zip(forwards, rows, discfactors, quantities, strict=True)])
    panels: list[_Panel] = []
    # A refined handle retains the original cumulative work, including discarded parents.
    evaluations = discarded_evaluations
    refinements = initial_refinements
    resolution = np.full(strikes.shape, np.inf)
    tail = np.full(strikes.shape, np.inf)

    def diagnostics(failed_expiry=None) -> QuadratureDiagnostics:
        """Report the quote with the largest error relative to its budget."""
        score = np.maximum(resolution, tail) if np.isfinite(tail).all() else resolution
        relative_error = score / budget
        if failed_expiry is not None:
            relative_error = np.where(expiry == failed_expiry, relative_error, -np.inf)
        worst = int(np.argmax(relative_error))
        return QuadratureDiagnostics(
            float(ttms[expiry[worst]]), float(strikes[worst]),
            float(panels[-1].right) if panels else 0.0,
            len(panels), evaluations, refinements, float(resolution[worst]), float(tail[worst]),
        )

    def sample(edges: np.ndarray, shells: Sequence[int]) -> list[_Panel]:
        """Evaluate chain transforms and embedded estimates on the requested panels."""
        nonlocal evaluations
        if evaluations + (edges.size - 1) * _NODES.size > max_nodes:
            needed = (edges.size - 1) * _NODES.size
            raise QuadratureError(f"frequency evaluation budget exhausted: need {needed} more, "
                                  f"limit {max_nodes}", diagnostics())
        failed_expiry = None
        try:
            if fixed_plan is None:
                nodes, fine, coarse = _map_panels(edges)
            else:
                nodes, fine, coarse = (a.reshape(-1, _NODES.size) for a in
                                       (fixed_plan.nodes, fixed_plan.weights,
                                        fixed_plan.embedded_weights))
            phi = real_phi + 1j * nodes.ravel()
            evaluations += phi.size
            transformed = iter(log_mgfs(phi))
            samples = np.empty((len(ttms), phi.size), dtype=complex)
            kernels = np.empty_like(samples)
            with np.errstate(over="raise", invalid="raise"):
                denominator = phi.imag**2 + 0.25
            for row in range(len(ttms)):
                failed_expiry = row
                values = np.asarray(next(transformed, None), dtype=complex)
                if values.shape != phi.shape:
                    raise ValueError("log transform needs one value per expiry and contour point")
                if not np.isfinite(values).all():
                    col = np.flatnonzero(~np.isfinite(values))[0]
                    frequency = phi[col].imag
                    name = "log transform" if quantities[row] == "price" else "differentiated numerator"
                    raise FloatingPointError(f"nonfinite {name} at frequency {frequency:g}")
                samples[row] = values
                with np.errstate(over="raise", invalid="raise", under="ignore"):
                    transform = np.exp(values) if quantities[row] == "price" else values
                    kernels[row] = transform / denominator
            if next(transformed, None) is not None:
                raise ValueError("unexpected extra expiry in log transforms")
            failed_expiry = None
            sampled = []
            panel_values = samples.reshape(len(ttms), *nodes.shape)
            panel_kernels = kernels.reshape(len(ttms), *nodes.shape)
            for idx, shell in enumerate(shells):
                values = panel_values[:, idx, :]
                u = nodes[idx]
                # The modulus is independent of the strike phase.
                with np.errstate(over="raise", invalid="raise", under="ignore"):
                    kernel = panel_kernels[:, idx, :]
                    phase = np.exp(-1j * log_moneyness[:, None] * u)
                    difference = (phase * kernel[expiry]).real @ (fine[idx] - coarse[idx])
                    error = premium_scale * np.abs(difference)
                    mass = premium_scale * (np.abs(kernel) @ fine[idx])[expiry]
                if not np.isfinite(error).all() or not np.isfinite(mass).all():
                    raise FloatingPointError("nonfinite premium integral")
                sampled.append(_Panel(edges[idx + 1], shell, values.copy(), error, mass))
            return sampled
        except FloatingPointError as exc:
            raise QuadratureError(str(exc), diagnostics(failed_expiry)) from exc

    # One batch for the starting partition, respecting both hard limits before solving.
    ends = [initial_cutoff]
    while (len(ends) < min(seed_panels, max_nodes // _NODES.size)
           and ends[-1] <= max_cutoff / 2.0):
        ends.append(2.0 * ends[-1])
    if fixed_plan is not None:
        initial_edges = fixed_plan.edges
    if initial_edges is None:
        panels = sample(np.array([0.0, *ends]), range(len(ends)))
    else:
        if initial_edges[-1] > max_cutoff:
            raise QuadratureError(
                f"validation cutoff budget exhausted: need {initial_edges[-1]:g}, "
                f"limit {max_cutoff:g}", diagnostics(),
            )
        if refinements > max_refinements:
            raise QuadratureError(f"validation refinement budget exhausted: need {refinements}, "
                                  f"limit {max_refinements}", diagnostics())
        # Shell boundaries are the same powers of two used during initial planning.
        shell_ends = [initial_cutoff]
        while shell_ends[-1] < initial_edges[-1]:
            shell_ends.append(2.0 * shell_ends[-1])
        shells = np.searchsorted(shell_ends, initial_edges[1:])
        panels = sample(initial_edges, shells)
    while True:
        errors = np.array([p.error for p in panels])
        resolution = errors.sum(axis=0)
        last = panels[-1]
        if last.shell >= 2:
            # Use entire dyadic shells even after the outermost panels were split.
            recent = [sum(p.mass for p in panels if p.shell == shell)
                      for shell in (last.shell - 1, last.shell)]
            tail = np.maximum(recent[0], recent[1])
        if np.any(resolution > budget):
            if fixed_plan is not None:
                raise QuadratureError("prepared grid requires panel refinement", diagnostics())
            if refinements >= max_refinements:
                raise QuadratureError("panel refinement budget exhausted", diagnostics())
            index = int(np.argmax(np.max(errors / budget, axis=1)))
            left = panels[index - 1].right if index else 0.0
            right = panels[index].right
            mid = left + (right - left) / 2.0
            children = sample(np.array([left, mid, right]), [panels[index].shell] * 2)
            panels[index:index + 1] = children
            refinements += 1
        elif np.any(tail > budget):
            if fixed_plan is not None:
                raise QuadratureError("prepared grid requires a wider cutoff", diagnostics())
            if last.right > max_cutoff / 2.0:
                raise QuadratureError("cutoff budget exhausted", diagnostics())
            panels.extend(sample(np.array([last.right, 2.0 * last.right]), [last.shell + 1]))
        else:
            break

    plan = fixed_plan or FourierPlan(np.array([0.0, *(p.right for p in panels)]))
    try:
        samples = np.concatenate([p.samples for p in panels], axis=1)
        prices = []
        for values, f, (k, types), d, quantity in zip(samples, forwards, rows, discfactors, quantities):
            if quantity == "price":
                prices.append(plan.price_vanillas(values, forward=f, strikes=k, optiontypes=types,
                                                  discfactor=d, is_spot_measure=is_spot_measure))
            else:
                result = plan.vanilla_sensitivities(values, forward=f, strikes=k, discfactor=d,
                                                    quantity=quantity)
                if quantity == "delta":
                    result[np.isin(types, ("C", "IC"))] += d  # derivative of the additive D*F call term
                prices.append(result)
    except FloatingPointError as exc:
        raise QuadratureError(str(exc), diagnostics()) from exc
    return plan, prices, diagnostics()
