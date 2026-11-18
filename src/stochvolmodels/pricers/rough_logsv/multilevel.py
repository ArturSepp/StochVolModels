"""Bounded sampling, discrete classical controls and independent-scramble intervals.

The allocation is explicit and fixed before sampling. Refinement streams, ordinary
raw-forward streams and optimization/confirmation phases have distinct spawn keys.
Intervals are pointwise; passing these checks does not qualify a parameter domain.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numba import njit
from scipy.special import ndtri
from scipy.stats import qmc, t

from stochvolmodels.pricers.rough_logsv.history import (
    conditional_payoffs,
    prepare_history,
    raw_forward,
    simulate_history,
)
from stochvolmodels.pricers.rough_logsv.params import RoughKernel, RoughLogSvParams


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class RoughMCConfig:
    """Frozen multilevel allocation; counts are antithetic pairs per scramble.

    The default allocation is a small research run, not a qualification claim.
    Use a separate phase for optimization and confirmation; phase is part of
    the reproducible stream identity. Sampling is serial and memory is bounded
    by chunk_pairs and max_tape_bytes. No adaptive budget increases occur.
    """

    grids: tuple = (32, 64)
    pairs: tuple = (128, 64)
    replicates: int = 8
    raw_pairs: int = 128
    bridge_steps_per_day: int = 16
    chunk_pairs: int = 32
    entropy: tuple = (20261118, 3101)
    phase: int = 1
    max_seconds: float = 600.0
    max_tape_bytes: int = 128 * 2**20

    def __post_init__(self):
        """Validate a fixed nested grid and independent-scramble allocation."""
        for name in ("grids", "pairs", "entropy"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("replicates", "raw_pairs", "bridge_steps_per_day", "chunk_pairs", "phase"):
            _positive_integer(getattr(self, name), name)
        if self.replicates < 2 or len(self.grids) < 2 or len(self.grids) != len(self.pairs):
            raise ValueError("At least two levels and two independent scrambles required")
        for value in self.grids + self.pairs:
            _positive_integer(value, "grid/pair count")
        if self.grids[0] % self.bridge_steps_per_day or any(
            b != 2 * a for a, b in zip(self.grids, self.grids[1:])
        ):
            raise ValueError("Nested grids must double and include the bridge grid")
        if any(n & (n - 1) for n in self.pairs + (self.raw_pairs, self.chunk_pairs)):
            raise ValueError("Sobol pair and chunk counts must be powers of two")
        if not self.entropy or any(not isinstance(x, int) or x < 0 for x in self.entropy):
            raise ValueError("Explicit nonnegative integer seed entropy required")
        if not np.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("A positive sampling time cap is required")
        _positive_integer(self.max_tape_bytes, "max_tape_bytes")

    def sequence(self, level, replicate, block, kind):
        """Return one deterministic stream identity, including the sampling phase."""
        return np.random.SeedSequence(
            self.entropy, spawn_key=(self.phase, level, replicate, block, kind)
        )


@dataclass(frozen=True)
class DiscretePdeControl:
    """Independently validated mean of the SAME discrete classical control.

    Means and absolute allowances are in forward-normalized premium units.
    validation_errors contains independently computed space/quadrature/domain/
    contour differences, one row per quote. Continuous PDE means alone cannot
    replace these discrete means. The quote ordering and control parameters are
    bound to this object; callers must retain the supporting validation evidence.
    """

    params: RoughLogSvParams
    days: tuple
    expiry_indices: tuple
    strikes_over_forward: tuple
    is_call: tuple
    means: tuple
    allowance: tuple
    validation_errors: tuple
    steps_per_day: int = 16

    def __post_init__(self):
        """Reject missing, mismatched or failed mean-validation evidence."""
        for name in (
            "days",
            "expiry_indices",
            "strikes_over_forward",
            "is_call",
            "means",
            "allowance",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(
            self, "validation_errors", tuple(tuple(row) for row in self.validation_errors)
        )
        if not isinstance(self.params, RoughLogSvParams):
            raise TypeError("Control parameters must be RoughLogSvParams")
        self.params.__post_init__()
        _positive_integer(self.steps_per_day, "control steps_per_day")
        n = len(self.means)
        a, m, errors = (
            np.asarray(x, float) for x in (self.allowance, self.means, self.validation_errors)
        )
        if (
            self.params.H != 0.5
            or self.params.martingale_margin < 0
            or n == 0
            or m.shape != (n,)
            or a.shape != (n,)
            or errors.ndim != 2
            or errors.shape[0] != n
            or errors.shape[1] < 2
            or not np.isfinite(m).all()
            or not np.isfinite(a).all()
            or np.any(a < 0)
            or not np.isfinite(errors).all()
            or np.any(np.abs(errors) > a[:, None])
        ):
            raise ValueError("Independently validated discrete control mean required")
        if any(len(v) != n for v in (self.expiry_indices, self.strikes_over_forward, self.is_call)):
            raise ValueError("Control means and quote coordinates must have the same length")
        if not self.days:
            raise ValueError("Control expiries must be positive increasing integer days")
        for day in self.days:
            _positive_integer(day, "control expiry")
        indices = np.asarray(self.expiry_indices)
        strikes = np.asarray(self.strikes_over_forward, float)
        if (
            np.any(np.diff(self.days) <= 0)
            or indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= len(self.days))
            or not np.isfinite(strikes).all()
            or np.any(strikes <= 0)
            or any(not isinstance(x, (bool, np.bool_)) for x in self.is_call)
        ):
            raise ValueError("Invalid discrete-control quote coordinates")
        object.__setattr__(self, "_parameter_values", tuple(self.params.to_dict().values()))

    def validate_for(self, days, expiry_indices, strikes, calls, config):
        """Check control-mean binding before any sampling starts."""
        # This comparison also catches a mutable parameter object changed after validation.
        if tuple(self.params.to_dict().values()) != self._parameter_values:
            raise ValueError("Control parameters changed after mean validation")
        self.__post_init__()
        for expected, supplied in (
            (self.days, days),
            (self.expiry_indices, expiry_indices),
            (self.strikes_over_forward, strikes),
            (self.is_call, calls),
        ):
            if not np.array_equal(expected, supplied):
                raise ValueError("Control mean belongs to different quote coordinates")
        if config.bridge_steps_per_day != self.steps_per_day:
            raise ValueError("Control mean was validated at a different time step")


@njit(cache=False)
def brownian_bridge(z):
    """Map independent standard normals to the frozen breadth-first Brownian bridge."""
    paths, n = z.shape
    W = np.zeros((n + 1, paths))
    W[n] = np.sqrt(n) * z[:, 0]
    left = np.empty(n, np.int64)
    right = np.empty(n, np.int64)
    left[0] = 0
    right[0] = n
    read = 0
    write = 1
    column = 1
    while read < write:
        lo = left[read]
        hi = right[read]
        read += 1
        if hi - lo <= 1:
            continue
        mid = (lo + hi) // 2
        W[mid] = ((hi - mid) * W[lo] + (mid - lo) * W[hi]) / (hi - lo) + np.sqrt(
            (mid - lo) * (hi - mid) / (hi - lo)
        ) * z[:, column]
        column += 1
        if mid - lo > 1:
            left[write] = lo
            right[write] = mid
            write += 1
        if hi - mid > 1:
            left[write] = mid
            right[write] = hi
            write += 1
    return W[1:] - W[:-1]


def aggregate_normal(z, grid, coarser_grid):
    """Sum nested Brownian increments without changing antithetic pairing."""
    if grid % coarser_grid:
        raise ValueError("Nested Brownian grids required")
    block = grid // coarser_grid
    return z if block == 1 else z.reshape(-1, block, z.shape[1]).sum(axis=1) / np.sqrt(block)


def make_sobol(config, days, level, replicate):
    """Create the paper's scrambled bridge net for one independent replicate."""
    dimension = int(days[-1]) * config.bridge_steps_per_day
    return qmc.Sobol(
        dimension,
        scramble=True,
        bits=52,
        seed=np.random.default_rng(config.sequence(level, replicate, 0, 0)),
    )


def normal_block(net, config, days, level, replicate, block, grid, pairs):
    """Generate the exact frozen bridge/refinement tape for one small batch."""
    if 8 * int(days[-1]) * grid * pairs * 6 > config.max_tape_bytes:
        raise MemoryError(
            "Brownian workspace exceeds configured max_tape_bytes; reduce chunk_pairs"
        )
    u = net.random(pairs)
    if np.any(u <= 0) or np.any(u >= 1):
        raise FloatingPointError("Sobol uniforms must be strictly inside (0,1)")
    coarse = brownian_bridge(ndtri(u))
    z = np.random.default_rng(config.sequence(level, replicate, block, 1)).normal(
        size=(int(days[-1]) * grid, pairs)
    )
    cells = z.reshape(
        int(days[-1]) * config.bridge_steps_per_day, grid // config.bridge_steps_per_day, pairs
    )
    cells -= cells.mean(axis=1, keepdims=True)
    cells += coarse[:, None, :] / np.sqrt(grid // config.bridge_steps_per_day)
    return np.ascontiguousarray(np.concatenate([z, -z], axis=1))


def prices_from_states(states, rho, expiry_indices, strikes, calls):
    """Map full path statistics to quotes in their original order."""
    index = np.asarray(expiry_indices, dtype=int)
    return conditional_payoffs(
        states[0, index],
        states[1, index],
        rho,
        np.asarray(strikes)[:, None],
        np.asarray(calls)[:, None],
    )


def summarize_replicates(levels, smoothing, forwards, allowance):
    """Reconstruct means and pointwise t intervals from independent scrambles.

    Each level has shape (scrambles, quotes). Variances add across independent
    levels; paths within a Sobol scramble are never treated as independent.
    The deterministic control-mean allowance is added to the price halfwidth.
    """
    levels = np.asarray(levels, float)
    smoothing, forwards = np.asarray(smoothing, float), np.asarray(forwards, float)
    if (
        levels.ndim != 3
        or levels.shape[0] < 2
        or levels.shape[1] < 2
        or smoothing.shape != levels.shape[1:]
        or forwards.ndim != 2
        or forwards.shape[0] != levels.shape[1]
        or not all(np.isfinite(x).all() for x in (levels, smoothing, forwards))
    ):
        raise ValueError("Complete finite level, smoothing and raw-forward replicates required")
    reps = levels.shape[1]
    a = np.broadcast_to(np.asarray(allowance, float), (levels.shape[2],))
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError("Nonnegative finite control allowance required")
    quantile = t.ppf(0.975, reps - 1)
    se = np.sqrt(np.sum(levels.var(axis=1, ddof=1) / reps, axis=0))
    return dict(
        price=levels.mean(axis=1).sum(axis=0),
        standard_error=se,
        price_halfwidth=quantile * se + a,
        time_change=levels[-1].mean(axis=0),
        time_halfwidth=quantile * levels[-1].std(axis=0, ddof=1) / np.sqrt(reps),
        smoothing_change=smoothing.mean(axis=0),
        smoothing_halfwidth=quantile * smoothing.std(axis=0, ddof=1) / np.sqrt(reps),
        raw_forward=forwards.mean(axis=0),
        raw_forward_halfwidth=quantile * forwards.std(axis=0, ddof=1) / np.sqrt(reps),
    )


def numerical_checks(
    summary,
    spreads,
    price_limit=0.1,
    component_limit=0.1,
    forward_halfwidth_limit=0.0005,
    forward_total_limit=0.001,
):
    """Apply pointwise paper thresholds; never open a parameter-domain calibration gate."""
    spreads = np.asarray(spreads, float)
    if (
        spreads.shape != summary["price"].shape
        or not np.isfinite(spreads).all()
        or np.any(spreads <= 0)
    ):
        raise ValueError("A positive full bid-ask spread is required for every quote")
    price = summary["price_halfwidth"] / spreads <= price_limit
    time = (abs(summary["time_change"]) + summary["time_halfwidth"]) / spreads <= component_limit
    smoothing = (
        abs(summary["smoothing_change"]) + summary["smoothing_halfwidth"]
    ) / spreads <= component_limit
    forward = (summary["raw_forward_halfwidth"] <= forward_halfwidth_limit) & (
        abs(summary["raw_forward"] - 1) + summary["raw_forward_halfwidth"] <= forward_total_limit
    )
    return dict(
        price=price,
        time=time,
        smoothing=smoothing,
        raw_forward=forward,
        pointwise_passed=bool(price.all() and time.all() and smoothing.all() and forward.all()),
        calibration_ready=False,
        parameter_domain_qualified=False,
    )


def sample_multilevel(
    params, kernel, comparison_kernel, config, control, days, expiry_indices, strikes, calls
):
    """Run a bounded serial multilevel experiment with a separate raw-forward sample.

    The validated control mean is checked before preparation or tape generation.
    The result retains per-level independent replicate means for replay/audit.
    """
    control.validate_for(days, expiry_indices, strikes, calls, config)
    if comparison_kernel.H != kernel.H or (comparison_kernel == kernel and kernel.H < 0.5):
        raise ValueError("An independent smoothing kernel comparison is required")
    start = perf_counter()
    preps = {g: prepare_history(params, kernel, 1 / (365 * g)) for g in config.grids}
    raw = prepare_history(params, kernel, 1 / (365 * config.grids[-1]), tilt=0)
    sm = prepare_history(params, comparison_kernel, 1 / (365 * config.grids[-1]))
    cv = prepare_history(control.params, RoughKernel.classical(), 1 / (365 * control.steps_per_day))
    levels = np.zeros((len(config.grids), config.replicates, len(strikes)))
    smooth = np.zeros((config.replicates, len(strikes)))
    forwards = np.zeros((config.replicates, len(days)))
    minimum = np.inf
    for level, count in enumerate(config.pairs + (config.raw_pairs,)):
        grid = config.grids[min(level, len(config.grids) - 1)]
        for rep in range(config.replicates):
            net = make_sobol(config, days, level, rep)
            for block, begin in enumerate(range(0, count, config.chunk_pairs)):
                if perf_counter() - start > config.max_seconds:
                    raise TimeoutError("Fixed sampling budget exhausted; calibration remains gated")
                pairs = min(config.chunk_pairs, count - begin)
                z = normal_block(net, config, days, level, rep, block, grid, pairs)
                stops = np.asarray(days) * grid
                if level == len(config.grids):
                    out, low = simulate_history(raw, z, stops)
                    forwards[rep] += raw_forward(out[0], out[1], params.rho).sum(axis=1) / (
                        2 * count
                    )
                else:
                    out, low = simulate_history(preps[grid], z, stops)
                    fine = prices_from_states(out, params.rho, expiry_indices, strikes, calls)
                    coarser = config.grids[level - 1] if level else control.steps_per_day
                    other, low2 = simulate_history(
                        preps[coarser] if level else cv,
                        aggregate_normal(z, grid, coarser),
                        np.asarray(days) * coarser,
                    )
                    low = min(low, low2)
                    difference = fine - prices_from_states(
                        other,
                        params.rho if level else control.params.rho,
                        expiry_indices,
                        strikes,
                        calls,
                    )
                    if level == 0:
                        difference += np.asarray(control.means)[:, None]
                    levels[level, rep] += difference.sum(axis=1) / (2 * count)
                    if level == len(config.grids) - 1:
                        other, low2 = simulate_history(sm, z, stops)
                        smooth[rep] += (
                            fine
                            - prices_from_states(other, params.rho, expiry_indices, strikes, calls)
                        ).sum(axis=1) / (2 * count)
                        low = min(low, low2)
                minimum = min(minimum, low)
    return dict(
        summary=summarize_replicates(levels, smooth, forwards, control.allowance),
        level_replicates=levels,
        smoothing_replicates=smooth,
        forward_replicates=forwards,
        minimum_sigma=minimum,
        seconds=perf_counter() - start,
        parameter_domain_qualified=False,
    )
