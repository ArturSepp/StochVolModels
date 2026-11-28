"""Coupled log-Euler path blocks on a synchronized equity observation grid."""

import numpy as np

from stochvolmodels.data.model_paths import ModelPaths
from stochvolmodels.models.lsv.simulation import LSVMonteCarloConfig, _normals


def monitoring_grid(model, fixings, steps_per_year=None):
    """Union of daily grid, native forward knots, volatility buckets and actual fixings."""
    end = max(fixings)
    interior = [model.curve_times, *(v.times for v in model.surfaces)]
    required = np.unique(np.r_[0.0, fixings])
    anchors = np.unique(np.concatenate(interior))
    regular = (
        np.arange(1, int(np.ceil(end * steps_per_year))) / steps_per_year
        if steps_per_year is not None
        else np.array([])
    )
    # Preserve exact contractual times. Near-identical arithmetic representations
    # of a calendar or curve knot would otherwise create unrefinable intervals.
    for candidates in (anchors, regular):
        candidates = candidates[(candidates > 0) & (candidates < end)]
        idx = np.searchsorted(required, candidates)
        distance = np.minimum(candidates - required[idx - 1], required[idx] - candidates)
        required = np.union1d(required, candidates[distance > 8 * np.spacing(max(1.0, end))])
    return required


def simulate_block(
    model,
    observation_times,
    config: LSVMonteCarloConfig,
    *,
    block_id: int,
    size: int,
    refinement: int = 0,
    first_group_id: int = 0,
) -> ModelPaths:
    """Generate correlated daily endpoints and cumulative frozen log variance.

    Each fixed logical block has a base stream and independent split streams.
    Each pair of refined Brownian increments sums to its parent increment.
    """
    base = np.asarray(observation_times, dtype=float)
    if base.ndim != 1 or len(base) < 2 or base[0] != 0 or not np.all(np.diff(base) > 0):
        raise ValueError("observation grid must increase from zero")
    if not np.all(np.isfinite(base)):
        raise ValueError("observation times must be finite")
    base = monitoring_grid(model, base)
    if not isinstance(refinement, int) or not 0 <= refinement <= 8:
        raise ValueError("refinement must be an integer from 0 to 8")
    if size < 1 or (config.antithetic and size % 2):
        raise ValueError("path block must contain complete independent groups")
    pieces = 2**refinement
    times = np.r_[
        np.concatenate(
            [np.linspace(a, b, pieces, endpoint=False) for a, b in zip(base[:-1], base[1:])]
        ),
        base[-1],
    ]
    log_fwd = model.log_forwards(times)
    rngs = [
        np.random.default_rng(np.random.SeedSequence(config.seed, spawn_key=(block_id, lvl)))
        for lvl in range(refinement + 1)
    ]
    n = len(model.asset_ids)
    x = np.zeros((size, n))
    assets = np.empty((size, len(times), n))
    variances = np.zeros_like(assets)
    assets[:, 0] = model.spots
    quanto = model.fx_vols * model.equity_fx_correlations

    def draws(rng):
        """Draw independent normal shocks per asset with the configured antithetic pairing."""
        return np.column_stack([_normals(rng, size, config.antithetic) for _ in range(n)])

    k = 0
    for _ in range(len(base) - 1):
        shocks = [draws(rngs[0])]
        for rng in rngs[1:]:
            children = []
            for parent in shocks:
                split = draws(rng)
                children.extend(((parent + split) / np.sqrt(2.0), (parent - split) / np.sqrt(2.0)))
            shocks = children
        for shock in shocks:
            dt = times[k + 1] - times[k]
            vols = np.column_stack(
                [law.evaluate(times[k] + dt / 2, x[:, j]) for j, law in enumerate(model.surfaces)]
            )
            variance = vols**2 * dt
            x += (shock @ model.root.T) * vols * np.sqrt(dt) - 0.5 * variance - vols * quanto * dt
            variances[:, k + 1] = variances[:, k] + variance
            assets[:, k + 1] = np.exp(log_fwd[k + 1] + x)
            k += 1
    if not np.all(np.isfinite(assets) & (assets > 0)) or not np.all(np.isfinite(variances)):
        raise FloatingPointError("nonfinite basket path or integrated variance")
    group_size = 2 if config.antithetic else 1
    group_ids = first_group_id + np.tile(np.arange(size // group_size), group_size)
    for data in (assets, variances, times, group_ids):
        data.setflags(write=False)
    return ModelPaths(
        observation_times=times,
        assets=assets,
        asset_ids=model.asset_ids,
        sampling_measure="Q_MMA",
        target_measure="Q_MMA",
        numeraire="money_market_account",
        scheme="basket_frozen_log_euler",
        states={"quadratic_variance": variances},
        state_units={"quadratic_variance": "integrated variance"},
        provenance={
            "asof": model.asof,
            "settlement_currency": model.settlement_currency,
            "seed": config.seed,
            "block_id": block_id,
            "refinement": refinement,
            "antithetic": config.antithetic,
            "independent_group_ids": group_ids,
            "asset_currencies": model.currencies,
            "fx_quotation": "settlement_per_native",
        },
    )
