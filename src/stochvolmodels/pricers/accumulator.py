"""Forward-density quotes and marks for discretely observed single-name accumulators.

The surviving law is independent of strike. Project the closing barrier once per
fixing, integrate the signed fixing at its payment discount, and reuse the law
in the par solve. Monetary results use settlement currency and original notional.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from stochvolmodels.models.basket import CorrelatedLocalVolModel
from stochvolmodels.pde_solvers import (
    BoundaryPair1D,
    Grid1D,
    PDE1DSolver,
    PDEDirection,
    TimeStepConfig,
)
from stochvolmodels.products.accumulator import (
    AccumulatorDirection,
    AccumulatorState,
    AccumulatorTerms,
)
from stochvolmodels.pde_solvers._prepared_forward import PreparedForwardSteps


class AccumulatorStatus(StrEnum):
    """Outcome of a quote or post-close mark."""

    OK = "ok"
    FIXED = "fixed"
    SETTLED = "settled"
    UNRESOLVED_HISTORY = "unresolved_history"
    ZERO_EXPOSURE = "zero_exposure"
    UNCERTIFIED = "uncertified"


@dataclass(frozen=True, kw_only=True)
class AccumulatorPdeConfig:
    """Initial log-forward mesh and controls for successive refined valuations.

    One nominal substep per interval is replaced by two implicit half-steps.
    The default grid is centered on current log-forward moneyness zero.
    """

    nodes: int = 1601
    half_width: float = 1.2
    substeps: int = 4
    max_refinements: int = 2
    pv_tolerance_bp: float = 0.25
    strike_tolerance_bp: float = 0.5
    mass_tolerance: float = 1e-9
    boundary_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        for name, minimum in (("nodes", 5), ("substeps", 1), ("max_refinements", 1)):
            n = getattr(self, name)
            if type(n) is not int or n < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.nodes % 2 == 0:
            raise ValueError("nodes must be odd to represent the initial point mass")
        for name in (
            "half_width",
            "pv_tolerance_bp",
            "strike_tolerance_bp",
            "mass_tolerance",
            "boundary_tolerance",
        ):
            x = getattr(self, name)
            if not np.isfinite(x) or x <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class AccumulatorDiagnostics:
    """Final mesh, per-close survival and successive-mesh discrepancies in basis points."""

    nodes: int
    half_width: float
    substeps: int
    refinements: int
    survival: tuple[float, ...]
    peak_negative_mass: float
    peak_boundary_mass: float
    peak_mass_error: float
    pv_change_bp: float
    strike_change_bp: float | None
    failure_reason: str | None = None


@dataclass(frozen=True, kw_only=True)
class AccumulatorValue:
    """Post-close PV split into unobserved fixings and earned, unpaid cash."""

    value: float
    future_fixings_pv: float
    fixed_unpaid_pv: float
    currency: str
    asof: str
    status: AccumulatorStatus
    diagnostics: AccumulatorDiagnostics | None = None


@dataclass(frozen=True, kw_only=True)
class AccumulatorQuote:
    """Issue strike divided by original spot, and signed issue PV at the solved strike."""

    strike_ratio: float
    par_residual: float
    currency: str
    asof: str
    status: AccumulatorStatus
    diagnostics: AccumulatorDiagnostics | None


def _discounts(terms, state, model, factors, currency):
    if currency != terms.currency:
        raise ValueError("discount and settlement currency must agree")
    if model is not None:
        if model.asof != state.asof or model.settlement_currency != currency:
            raise ValueError("model and state asof and settlement currency must agree")
        if terms.asset_id not in model.asset_ids:
            raise ValueError("model is missing the contractual asset")
    indices = state.outstanding_payments(terms)
    discounts = np.asarray(factors, dtype=float)
    if discounts.shape != indices.shape or not np.all(np.isfinite(discounts) & (discounts > 0)):
        raise ValueError("require one positive finite discount per outstanding payment")
    result = np.zeros(len(terms.payment_times))
    result[indices] = discounts
    return result


def _surviving_law(terms, state, model, discounts, nodes, width, substeps, *, prepare=False):
    idx = model.asset_ids.index(terms.asset_id)
    provider = model.marginal_pde_coefficient_provider(terms.asset_id)
    grid = Grid1D.uniform(-width, width, nodes)
    edges = grid.cell_edges
    cell_width = np.diff(edges)
    weights = grid.quadrature_weights
    first = state.next_fixing(terms)
    fixings = terms.fixing_times[first:] - state.elapsed
    forwards = model.log_forwards(fixings)[:, idx]
    log_barriers = np.log(terms.barrier) - forwards
    levels = np.exp(forwards[:, None] + grid.nodes)
    if not np.all(np.isfinite(levels) & (levels > 0)):
        raise FloatingPointError("log-forward domain exceeds representable spot levels")
    discounted = np.empty_like(levels)
    survival = []
    negative = boundary = mass_error = 0.0
    previous_mass = 1.0
    density = grid.project_delta(0.0).copy()
    implicit = PDE1DSolver(TimeStepConfig(theta=1.0))
    centred = PDE1DSolver(TimeStepConfig(theta=0.5))
    boundaries = BoundaryPair1D.natural()
    prepared = (
        PreparedForwardSteps(grid, provider, boundaries)
        if prepare and not provider.is_time_dependent else None
    )
    knots = np.union1d(model.curve_times, model.surfaces[idx].times)
    start = 0.0
    for j, end in enumerate(fixings):
        count = substeps * max(1, int(np.ceil((end - start) * 252 - 1e-10)))
        mesh = np.linspace(start, end, count + 1)
        interior = knots[(knots > start) & (knots < end)]
        # Ignore arithmetic duplicates but retain the original fixing endpoints.
        loc = np.searchsorted(mesh, interior)
        extra = interior[np.minimum(interior - mesh[loc - 1], mesh[loc] - interior) > 1e-12]
        mesh = np.union1d(mesh, extra)
        damping = (start, start + (mesh[1] - start) * 0.5, mesh[1])
        for solver, segment in ((implicit, damping), (centred, mesh[1:])):
            if len(segment) < 2:
                continue
            if prepared is None:
                solution = solver.solve(
                    grid, provider, boundaries, PDEDirection.FORWARD, segment, density
                )
                density = solution.values
                negative = max(negative, *(d.negative_mass for d in solution.diagnostics))
            else:
                negative = max(negative, prepared.advance(segment, solver.config.theta, density))
            mass_error = max(mass_error, abs(grid.integrate(density) - previous_mass))
            boundary = max(boundary, float(np.abs(density[[0, -1]]) @ weights[[0, -1]]))
        keep = np.clip((log_barriers[j] - edges[:-1]) / cell_width, 0, 1)
        if terms.direction is AccumulatorDirection.DECUMULATOR:
            keep = 1 - keep
        density = density * keep
        mass = density * weights
        remaining = float(np.sum(mass))
        mass_error = max(mass_error, remaining - previous_mass, -remaining)
        survival.append(remaining)
        discounted[j] = discounts[terms.payment_indices[first + j]] * mass
        previous_mass = remaining
        start = end

    def pv(strike):
        return float(np.sum(discounted * terms.fixing_cash(levels, strike)))

    return (
        pv,
        (float(levels.min()), float(levels.max())),
        tuple(survival),
        negative,
        boundary,
        mass_error,
    )


def _calculate(terms, state, model, discounts, config, locked_strike):
    nodes, width, steps = config.nodes, config.half_width, config.substeps
    previous_pv = previous_strike = None
    is_quote = locked_strike is None
    for level in range(config.max_refinements + 1):
        try:
            with np.errstate(over="raise", invalid="raise"):
                pv, bracket, survival, negative, boundary, mass_error = _surviving_law(
                    terms, state, model, discounts, nodes, width, steps, prepare=True
                )
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            diagnostics = AccumulatorDiagnostics(
                nodes,
                width,
                steps,
                level,
                (),
                np.nan,
                np.nan,
                np.nan,
                np.inf,
                np.inf if is_quote else None,
                str(exc),
            )
            return np.nan, np.nan, AccumulatorStatus.UNCERTIFIED, diagnostics
        healthy = (
            negative <= config.mass_tolerance
            and mass_error <= config.mass_tolerance
            and boundary <= config.boundary_tolerance
        )
        strike = locked_strike
        if is_quote:
            if not any(m > 0 for m in survival) and healthy:
                return np.nan, np.nan, AccumulatorStatus.ZERO_EXPOSURE, None
            sign = 1 if terms.direction is AccumulatorDirection.ACCUMULATOR else -1
            bounds = np.asarray(bracket) / terms.original_spot
            if sign * pv(bounds[0]) >= 0 >= sign * pv(bounds[1]):
                strike = brentq(pv, *bounds, xtol=1e-13)
            else:
                healthy = False
        value = np.nan if strike is None else pv(strike)
        pv_delta = np.inf
        strike_delta = None if not is_quote else np.inf
        if previous_pv is not None and strike is not None:
            # Compare both meshes at the same strike, even for an issue quote.
            pv_delta = abs(value - previous_pv(strike)) * 1e4 / terms.notional
            if is_quote and previous_strike is not None:
                strike_delta = abs(strike - previous_strike) * 1e4
        diag = AccumulatorDiagnostics(
            nodes,
            width,
            steps,
            level,
            survival,
            negative,
            boundary,
            mass_error,
            pv_delta,
            strike_delta,
        )
        if (
            healthy
            and pv_delta <= config.pv_tolerance_bp
            and (
                not is_quote
                or (
                    strike_delta <= config.strike_tolerance_bp
                    and abs(value) <= terms.notional * 1e-5
                )
            )
        ):
            return strike, value, AccumulatorStatus.OK, diag
        previous_pv, previous_strike = pv, strike
        # Hold or improve spatial resolution when widening the domain.
        growth = 1.0
        if boundary > config.boundary_tolerance:
            # A steep fitted wing can need several times the flat-reference domain.
            # Use the marginal's diffusion and absolute drift envelope to choose
            # the trial width. Acceptance still depends on measured boundary mass.
            idx = model.asset_ids.index(terms.asset_id)
            sigma = float(np.max(model.surfaces[idx].values))
            horizon = terms.fixing_times[-1] - state.elapsed
            loading = abs(model.fx_vols[idx] * model.equity_fx_correlations[idx])
            trial_width = 7 * sigma * np.sqrt(horizon) + horizon * sigma * (0.5 * sigma + loading)
            growth = float(np.clip(trial_width / width, 1.5, 3.5))
        nodes = 2 * int(np.ceil((nodes - 1) / 2 * np.sqrt(2) * growth)) + 1
        width *= growth
        steps *= 2
    return np.nan, np.nan, AccumulatorStatus.UNCERTIFIED, diag


def quote_accumulator_strike(
    terms: AccumulatorTerms,
    model: CorrelatedLocalVolModel,
    *,
    discount_factors: ArrayLike,
    discount_currency: str,
    config: AccumulatorPdeConfig = AccumulatorPdeConfig(),
) -> AccumulatorQuote:
    """Solve a zero-premium issue strike, expressed as a ratio to original spot.

    Parameters
    ----------
    terms : AccumulatorTerms
        Original contract with fixing and payment times measured from issue.
    model : CorrelatedLocalVolModel
        Issue-date market with native spot equal to the original fixing.
    discount_factors : array_like
        Settlement-currency discounts to distinct payment dates in increasing order.
    discount_currency : str
        Currency of the supplied discounts.
    config : AccumulatorPdeConfig
        Initial mesh and numerical acceptance limits.

    Returns
    -------
    AccumulatorQuote
        Strike ratio, signed par residual and diagnostics. A rejected quote has a NaN strike.
    """
    state = AccumulatorState.at_issue(terms, model.asof)
    discounts = _discounts(terms, state, model, discount_factors, discount_currency)
    if not np.isclose(
        model.spots[model.asset_ids.index(terms.asset_id)], terms.original_spot, rtol=1e-12, atol=0
    ):
        raise ValueError("issue quote requires original spot equal to current model spot")
    strike, pv, status, diagnostics = _calculate(terms, state, model, discounts, config, None)
    return AccumulatorQuote(
        strike_ratio=strike,
        par_residual=pv,
        currency=terms.currency,
        asof=state.asof,
        status=status,
        diagnostics=diagnostics,
    )


def value_accumulator(
    terms: AccumulatorTerms,
    state: AccumulatorState,
    model: CorrelatedLocalVolModel | None,
    *,
    strike_ratio: float,
    discount_factors: ArrayLike,
    discount_currency: str,
    config: AccumulatorPdeConfig = AccumulatorPdeConfig(),
) -> AccumulatorValue:
    """Mark the original locked strike after the as-of close and all payments due.

    Parameters
    ----------
    terms : AccumulatorTerms
        Original contract including its full schedule and original fixing spot.
    state : AccumulatorState
        Post-close evidence, elapsed ACT/365 time and fixed unpaid amounts.
    model : CorrelatedLocalVolModel or None
        Current market for future fixings. Deterministic cash accepts None.
    strike_ratio : float
        Locked strike divided by the original spot.
    discount_factors : array_like
        Current discounts in the order of state.outstanding_payments(terms).
    discount_currency : str
        Currency of the supplied discounts.
    config : AccumulatorPdeConfig
        Initial mesh and numerical acceptance limits.

    Returns
    -------
    AccumulatorValue
        Signed total and component PVs in settlement currency, status and diagnostics.
    """
    if not np.isfinite(strike_ratio) or strike_ratio <= 0:
        raise ValueError("strike_ratio must be positive and finite")
    discounts = _discounts(terms, state, model, discount_factors, discount_currency)
    fixed = float(state.fixed_unpaid @ discounts)
    diag = None
    if not state.history_complete:
        future = (
            0.0 if state.stopped or state.next_fixing(terms) == len(terms.fixing_times) else np.nan
        )
        fixed, status = np.nan, AccumulatorStatus.UNRESOLVED_HISTORY
    elif state.stopped or state.next_fixing(terms) == len(terms.fixing_times):
        future = 0.0
        status = (
            AccumulatorStatus.FIXED if np.any(state.fixed_unpaid) else AccumulatorStatus.SETTLED
        )
    else:
        if model is None:
            raise ValueError("a live stochastic position requires a market model")
        _, future, status, diag = _calculate(terms, state, model, discounts, config, strike_ratio)
    return AccumulatorValue(
        value=fixed + future,
        future_fixings_pv=future,
        fixed_unpaid_pv=fixed,
        currency=terms.currency,
        asof=state.asof,
        status=status,
        diagnostics=diag,
    )
