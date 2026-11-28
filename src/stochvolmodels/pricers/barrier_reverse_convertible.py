"""Cash-settled worst-of BRC pricing in the settlement money-market measure.

Quote the unconditional maturity coupon at par or mark one or more fixed coupons.
Use daily correlated paths and bracket the synchronized continuous knock-in with
conditional Frechet bounds. All monetary PVs are in the terms' settlement currency.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from hashlib import sha256

import numpy as np

from stochvolmodels.models.basket.model import CorrelatedLocalVolModel
from stochvolmodels.models.basket.simulation import monitoring_grid, simulate_block
from stochvolmodels.models.lsv.simulation import LSVMonteCarloConfig
from stochvolmodels.pricers._barrier_monitoring import barrier_hit_weights, bridge_inputs
from stochvolmodels.products.barrier_reverse_convertible import (
    BrcState,
    BrcTerms,
    Monitoring,
    TouchState,
)
from stochvolmodels.valuation import DiscountedMcEstimate, reduce_discounted_mc


@dataclass(frozen=True, kw_only=True)
class BrcConfig:
    """Simulation settings, logical block size in groups and dependence budget in face bp."""

    mc: LSVMonteCarloConfig = field(default_factory=lambda: LSVMonteCarloConfig(steps_per_year=252))
    groups_per_block: int = 4096
    dependence_budget_bp: float = 1.0
    initial_refinement: int = 0
    max_refinement: int = 1
    keep_groups: bool = False

    def __post_init__(self):
        if not isinstance(self.groups_per_block, int) or self.groups_per_block < 1:
            raise ValueError("groups_per_block must be positive")
        if not (
            isinstance(self.initial_refinement, int)
            and isinstance(self.max_refinement, int)
            and 0 <= self.initial_refinement <= self.max_refinement <= 8
        ):
            raise ValueError("invalid refinement limits")
        if not np.isfinite(self.dependence_budget_bp) or self.dependence_budget_bp < 0:
            raise ValueError("invalid dependence budget")
        if self.mc.paths // (2 if self.mc.antithetic else 1) < 2:
            raise ValueError("at least two independent groups required")


@dataclass(frozen=True, kw_only=True)
class BrcRequest:
    """One fixed coupon and its payment-date discount on a shared market snapshot."""

    terms: BrcTerms
    state: BrcState
    coupon: float
    discount_factor: float
    discount_currency: str


class BrcStatus(StrEnum):
    """BRC result status, serialized with the original public strings."""

    OK = "ok"
    FIXED = "fixed"
    PAID = "paid"
    UNRESOLVED_HISTORY = "unresolved_history"
    DEPENDENCE_TOLERANCE_EXCEEDED = "dependence_tolerance_exceeded"


@dataclass(frozen=True, kw_only=True)
class BrcValue:
    """Cash PV and discounted redemption, plus undiscounted loss diagnostics per face."""

    value: float
    redemption_pv: float
    coupon_pv: float
    standard_error: float
    expected_loss: float
    loss_error: float
    hit_probability: float
    loss_bounds: tuple[float, float]
    width_upper_bp: float
    monitoring: Monitoring
    status: BrcStatus
    refinement: int
    n_groups: int
    asof: str
    seed: int | None = None
    steps_per_year: int = 0
    pv_groups: DiscountedMcEstimate | None = None
    loss_groups: DiscountedMcEstimate | None = None
    refinement_history: tuple[float, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "status", BrcStatus(self.status))


@dataclass(frozen=True)
class BrcQuote:
    """Par maturity coupon, error and bounds per face, with the zero-coupon redemption valuation."""

    coupon: float
    standard_error: float
    coupon_bounds: tuple[float, float]
    valuation: BrcValue

    @property
    def status(self) -> BrcStatus:
        """Status of the redemption valuation underlying this quote."""
        return self.valuation.status


def _validate(req, model):
    t, state = req.terms, req.state
    if t.currency != req.discount_currency or (
        model is not None and t.currency != model.settlement_currency
    ):
        raise ValueError("contract, discount and model settlement currency must agree")
    if model is not None and state.asof != model.asof:
        raise ValueError("state and model as-of mismatch")
    if model is not None and not set(t.asset_ids).issubset(model.asset_ids):
        raise ValueError("missing required asset IDs")
    if (
        not np.isfinite(req.coupon)
        or not np.isfinite(req.discount_factor)
        or req.discount_factor <= 0
    ):
        raise ValueError("invalid coupon or discount")
    if state.final_levels is not None and state.final_levels.shape != t.fixings.shape:
        raise ValueError("final levels must match the contract")
    if state.paid and t.payment_time > 0 or t.fixing_time > 0 and state.final_levels is not None:
        raise ValueError("future payment or fixing recorded as historical")
    if t.payment_time < 0 and not state.paid:
        raise ValueError("past payment must have an explicit paid state")


def _resolved(req, model):
    """Return deterministic or unresolved results before requesting paths."""
    t, st = req.terms, req.state
    status = "ok"
    if st.paid:
        loss, hit = 0.0, 0.0
    elif t.fixing_time <= 0:
        if st.final_levels is None:
            raise ValueError("require observed final levels after fixing")
        terminal_hit = bool(np.any(st.final_levels <= t.barriers))
        unresolved = (
            t.monitoring is Monitoring.SYNCHRONIZED
            and st.touch is TouchState.UNRESOLVED
            and not terminal_hit
        )
        hit = (
            np.nan
            if unresolved
            else float(
                terminal_hit
                or (t.monitoring is Monitoring.SYNCHRONIZED and st.touch is TouchState.HIT)
            )
        )
        shortfall = t.loss_at_touch(st.final_levels)
        loss = 0.0 if shortfall == 0 else hit * shortfall
        status = "unresolved_history" if unresolved and shortfall > 0 else "fixed"
    elif (
        t.monitoring is Monitoring.SYNCHRONIZED
        and st.touch is TouchState.UNRESOLVED
        and (
            model is None
            or not np.any(
                model.spots[[model.asset_ids.index(a) for a in t.asset_ids]] <= t.barriers
            )
        )
    ):
        loss, hit, status = np.nan, np.nan, "unresolved_history"
    else:
        return None
    df_face = req.discount_factor * t.face
    redemption = 0.0 if st.paid else df_face * (1 - loss)
    coupon = 0.0 if st.paid else df_face * req.coupon
    err = np.nan if status == "unresolved_history" else 0.0
    return BrcValue(
        value=redemption + coupon,
        redemption_pv=redemption,
        coupon_pv=coupon,
        standard_error=err,
        expected_loss=loss,
        loss_error=err,
        hit_probability=hit,
        loss_bounds=(loss, loss),
        width_upper_bp=err,
        monitoring=t.monitoring,
        status="paid" if st.paid else status,
        refinement=0,
        n_groups=0,
        asof=st.asof,
    )


def _monitor_indices(model, indices, barriers):
    """Collapse indistinguishable perfectly correlated normalized diffusion paths."""
    selected = []
    for j, i in enumerate(indices):
        duplicate = False
        for k in selected:
            h = indices[k]
            a, b = model.surfaces[i], model.surfaces[h]
            if (
                model.correlation[i, h] == 1
                and barriers[j] / model.spots[i] == barriers[k] / model.spots[h]
                and np.array_equal(a.times, b.times)
                and np.array_equal(a.grid.nodes, b.grid.nodes)
                and np.array_equal(a.values, b.values)
                and np.allclose(
                    model.forward_levels[:, i] / model.spots[i],
                    model.forward_levels[:, h] / model.spots[h],
                    rtol=1e-14,
                    atol=0,
                )
                and model.fx_vols[i] * model.equity_fx_correlations[i]
                == model.fx_vols[h] * model.equity_fx_correlations[h]
            ):
                duplicate = True
                break
        if not duplicate:
            selected.append(j)
    return indices[selected], barriers[selected]


def _block_metrics(paths, requests, model):
    """Accumulate pathwise loss, hit, bound width and signed PV on common endpoints."""
    if paths.provenance["settlement_currency"] != model.settlement_currency:
        raise ValueError("settlement currency differs on paths")
    if paths.provenance["asof"] != model.asof or paths.asset_ids != model.asset_ids:
        raise ValueError("path as-of and asset order must match the market snapshot")
    levels, var = bridge_inputs(paths)
    cache = {}
    for req in requests:
        t = req.terms
        idx = np.array([model.asset_ids.index(name) for name in t.asset_ids])
        end = int(np.searchsorted(paths.observation_times, t.fixing_time))
        if end == len(paths.observation_times) or paths.observation_times[end] != t.fixing_time:
            raise ValueError("paths must include each requested fixing")
        terminal = levels[:, end, :][:, idx]
        if t.monitoring is Monitoring.MATURITY:
            weights = np.broadcast_to(
                np.any(terminal <= t.barriers, axis=1)[:, None], (len(terminal), 3)
            )
        elif req.state.touch is TouchState.HIT or np.any(model.spots[idx] <= t.barriers):
            weights = np.ones((len(terminal), 3))
        else:
            key = (t.asset_ids, tuple(t.barriers), end)
            if key not in cache:
                active, barriers = _monitor_indices(model, idx, t.barriers)
                eligible = np.ones((end, len(active)), dtype=bool)
                cache[key] = barrier_hit_weights(levels, var, active, barriers, eligible)
            weights = cache[key]
        losses = weights * t.loss_at_touch(terminal)[:, None]
        yield np.column_stack(
            (
                losses,
                weights[:, 1],
                losses[:, 2] - losses[:, 0],
                req.discount_factor * t.face * (1 + req.coupon - losses[:, 1]),
            )
        )


def _price_resolution(requests, model, config, grid, level):
    mc = config.mc
    group_size = 2 if mc.antithetic else 1
    total = mc.paths // group_size
    stream = repr(
        (
            mc.seed,
            mc.antithetic,
            config.groups_per_block,
            model.asset_ids,
            sha256(grid.tobytes()).hexdigest(),
        )
    )
    # The first three sums describe loss bounds and hit probability. Keep only
    # the two requested group series when callers need cross-position errors.
    sums = np.zeros((len(requests), 3))
    means = np.zeros((len(requests), 3))
    moments = np.zeros_like(means)
    kept = np.empty((len(requests), 2, total)) if config.keep_groups else None
    for block, start in enumerate(range(0, total, config.groups_per_block)):
        count = min(config.groups_per_block, total - start)
        paths = simulate_block(
            model,
            grid,
            mc,
            block_id=block,
            size=count * group_size,
            refinement=level,
            first_group_id=start,
        )
        for i, metrics in enumerate(_block_metrics(paths, requests, model)):
            sums[i] += metrics[:, [0, 2, 3]].sum(axis=0) / group_size
            for j, column in enumerate((1, 5, 4)):
                part = reduce_discounted_mc(
                    metrics[:, column],
                    paths.provenance["independent_group_ids"],
                    group_size=group_size,
                    stream_id=stream,
                    keep_groups=kept is not None and j < 2,
                )
                delta = part.value - means[i, j]
                moments[i, j] += part.m2 + delta**2 * start * count / (start + count)
                means[i, j] += delta * count / (start + count)
                if kept is not None and j < 2:
                    kept[i, j, start : start + count] = part.group_values
        del paths
    results = []
    ids = np.arange(total)
    ids.setflags(write=False)
    if kept is not None:
        kept.setflags(write=False)
    for i, req in enumerate(requests):
        loss, pv, width = (
            DiscountedMcEstimate(
                means[i, j],
                moments[i, j],
                ids,
                group_size,
                stream,
                kept[i, j] if kept is not None and j < 2 else None,
            )
            for j in range(3)
        )
        upper_bp = 1e4 * (width.value + 1.6448536269514722 * width.standard_error)
        coupon_pv = req.discount_factor * req.terms.face * req.coupon
        results.append(
            BrcValue(
                value=pv.value,
                redemption_pv=pv.value - coupon_pv,
                coupon_pv=coupon_pv,
                standard_error=pv.standard_error,
                expected_loss=loss.value,
                loss_error=loss.standard_error,
                hit_probability=sums[i, 2] / total,
                loss_bounds=(sums[i, 0] / total, sums[i, 1] / total),
                width_upper_bp=upper_bp,
                monitoring=req.terms.monitoring,
                status="ok"
                if upper_bp <= config.dependence_budget_bp
                else "dependence_tolerance_exceeded",
                refinement=level,
                n_groups=total,
                asof=model.asof,
                seed=mc.seed,
                steps_per_year=mc.steps_per_year * 2**level,
                pv_groups=pv if config.keep_groups else None,
                loss_groups=loss if config.keep_groups else None,
            )
        )
    return results


def value_brc_batch(
    requests: Iterable[BrcRequest],
    model: CorrelatedLocalVolModel | None,
    *,
    config: BrcConfig = BrcConfig(),
) -> tuple[BrcValue, ...]:
    """Mark ordered fixed coupons on a shared basket under their individual discounts."""
    requests = tuple(requests)
    values = []
    live = []
    for i, req in enumerate(requests):
        _validate(req, model)
        value = _resolved(req, model)
        values.append(value)
        if value is None:
            live.append(i)
    if live:
        if model is None:
            raise ValueError("a live stochastic BRC requires a market model")
        grid = monitoring_grid(
            model, [requests[i].terms.fixing_time for i in live], config.mc.steps_per_year
        )
        history = {i: [] for i in live}
        for level in range(config.initial_refinement, config.max_refinement + 1):
            estimates = _price_resolution([requests[i] for i in live], model, config, grid, level)
            for i, value in zip(live, estimates):
                history[i].append(value.width_upper_bp)
                values[i] = replace(value, refinement_history=tuple(history[i]))
            live = [i for i in live if values[i].status != "ok"]
            if not live:
                break
    return tuple(values)


def value_brc(
    terms: BrcTerms,
    state: BrcState,
    model: CorrelatedLocalVolModel | None,
    *,
    coupon: float,
    discount_factor: float,
    discount_currency: str,
    config: BrcConfig = BrcConfig(),
) -> BrcValue:
    """Mark a fixed maturity coupon and disclose its separately discounted redemption."""
    req = BrcRequest(
        terms=terms,
        state=state,
        coupon=coupon,
        discount_factor=discount_factor,
        discount_currency=discount_currency,
    )
    return value_brc_batch((req,), model, config=config)[0]


def quote_brc_coupon(
    terms: BrcTerms,
    model: CorrelatedLocalVolModel,
    *,
    discount_factor: float,
    discount_currency: str,
    config: BrcConfig = BrcConfig(),
) -> BrcQuote:
    """Solve the par maturity coupon from expected loss for an issue struck at current spots."""
    idx = [model.asset_ids.index(a) for a in terms.asset_ids]
    if terms.fixing_time <= 0 or not np.allclose(
        terms.fixings, model.spots[idx], rtol=1e-12, atol=0
    ):
        raise ValueError("par quote requires future fixing and original fixings at current spots")
    base = value_brc(
        terms,
        BrcState(asof=model.asof, touch=TouchState.CLEAR),
        model,
        coupon=0.0,
        discount_factor=discount_factor,
        discount_currency=discount_currency,
        config=config,
    )
    funding = 1 / discount_factor - 1
    return BrcQuote(
        funding + base.expected_loss,
        base.loss_error,
        tuple(funding + x for x in base.loss_bounds),
        base,
    )
