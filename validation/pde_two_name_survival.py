"""Independent survival controls and one-factor-at-a-time PIDE refinements for the public case."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import njit
from scipy.special import gammainc
from scipy.stats import poisson

from examples.pde_solvers.run_two_name_survival import Payoff, load_case, prices


def poisson_gamma(model, maturities):
    """Mix independent Erlang losses conditional on common and individual arrival counts."""
    if any(model.drifts):
        raise ValueError("the monotone pure-jump control requires zero drift")
    rows = []
    for t in maturities:
        probabilities = []
        for rate in (model.common_intensity, *model.axis_intensities):
            mean = rate * t
            last = int(poisson.isf(1e-16, mean)) if mean else 0
            probabilities.append(poisson.pmf(np.arange(last + 1), mean))
        common, *individual = probabilities
        conditional = []
        for barrier, rate, own in zip(model.barriers, model.mark_rates, individual, strict=True):
            n = np.add.outer(np.arange(common.size), np.arange(own.size))
            cdf = np.ones(n.shape)
            cdf[n > 0] = gammainc(n[n > 0], -barrier * rate)
            conditional.append(cdf @ own)
        x, y = conditional
        rows.append((common @ x, common @ y, common @ (x * y)))
    return np.array(rows)


@njit
def _bridge_survival(start, end, var_dt):
    """Conditional probability of staying above zero for a scalar Brownian bridge."""
    if start <= 0 or end <= 0:
        return 0.0
    if var_dt == 0:
        return 1.0
    return -np.expm1(-2 * start * end / var_dt)


@njit
def _simulate(paths, seed, intervals, dates, barriers, rates, clocks, variance, rho):
    # Accumulate moments of (conditional path value minus its coupled pure-jump payoff).
    np.random.seed(seed)
    sums = np.zeros((len(dates), 5))
    squares = np.zeros_like(sums)
    total = clocks.sum()
    orthogonal = np.sqrt(1 - rho * rho)
    for _ in range(paths):
        x, y = -barriers[0], -barriers[1]
        pure_x, pure_y = x, y
        t = 0.0
        jump_time = np.random.exponential(1 / total) if total else np.inf
        sx = sy = lower = upper = 1.0
        date_index = 0
        for k in range(1, dates[-1] + 1):
            stop = k / intervals
            while t < stop:
                end = min(stop, jump_time)
                dt = end - t
                z1, z2 = np.random.normal(), np.random.normal()
                next_x = x + np.sqrt(variance[0] * dt) * z1
                next_y = y + np.sqrt(variance[1] * dt) * (rho * z1 + orthogonal * z2)
                bx = _bridge_survival(x, next_x, variance[0] * dt)
                by = _bridge_survival(y, next_y, variance[1] * dt)
                sx *= bx
                sy *= by
                if rho == 0.0:
                    lower *= bx * by
                    upper *= bx * by
                else:
                    lower *= max(0.0, bx + by - 1.0)
                    upper *= min(bx, by)
                x, y, t = next_x, next_y, end
                if t == jump_time:
                    event = np.random.random() * total
                    if event < clocks[0] or event >= clocks[0] + clocks[1]:
                        loss = np.random.exponential(1 / rates[0])
                        x -= loss
                        pure_x -= loss
                    if event >= clocks[0]:
                        loss = np.random.exponential(1 / rates[1])
                        y -= loss
                        pure_y -= loss
                    if x <= 0:
                        sx = 0.0
                    if y <= 0:
                        sy = 0.0
                    if x <= 0 or y <= 0:
                        lower = upper = 0.0
                    jump_time += np.random.exponential(1 / total)
            if k == dates[date_index]:
                cx = 1.0 if pure_x > 0 else 0.0
                cy = 1.0 if pure_y > 0 else 0.0
                q = (sx - cx, sy - cy, lower - cx * cy, upper - cx * cy, upper - lower)
                for j in range(5):
                    sums[date_index, j] += q[j]
                    squares[date_index, j] += q[j] * q[j]
                date_index += 1
    return sums, squares


def bridge_benchmark(model, maturities, paths, seed, bridge_intervals_per_year):
    """Bracket joint continuous survival using exact marginal crossing probabilities.

    Given all jump times, marks and Gaussian endpoints, bridges on disjoint intervals
    are independent. Within one interval Fréchet bounds bracket joint survival for
    correlated coordinates. Jump overshoots kill immediately. Coupling to the exact
    pure-jump control reduces sampling error without changing either bound's mean.
    """
    if any(model.drifts) or paths < 2 or bridge_intervals_per_year < 1:
        raise ValueError("expected zero drift, at least two paths and a positive bridge frequency")
    grid = np.asarray(maturities) * bridge_intervals_per_year
    dates = np.rint(grid).astype(np.int64)
    if not np.array_equal(grid, dates) or dates[0] < 1 or np.any(np.diff(dates) <= 0):
        raise ValueError("maturities must be increasing bridge-grid dates")
    control = poisson_gamma(model, maturities)
    sums, squares = _simulate(
        paths,
        seed,
        bridge_intervals_per_year,
        dates,
        np.asarray(model.barriers),
        np.asarray(model.mark_rates),
        np.array([*model.axis_intensities, model.common_intensity]),
        np.asarray(model.variances),
        model.correlation,
    )
    means = sums / paths
    se = np.sqrt(np.maximum(0, squares - paths * means**2) / (paths * (paths - 1)))
    means[:, :4] += control[:, [0, 1, 2, 2]]
    return {
        name: {"estimate": means[:, j].tolist(), "se": se[:, j].tolist()}
        for j, name in enumerate(("x", "y", "joint_lower", "joint_upper", "monitoring_width"))
    }


def _observables(q):
    return {**q, "first_default": 1 - q["joint"], "both_default": 1 - q["x"] - q["y"] + q["joint"]}


def _valid(q):
    if np.any(q["joint"] > np.minimum(q["x"], q["y"]) + 1e-12):
        raise AssertionError("joint survival exceeds a marginal")
    for name, values in _observables(q).items():
        if np.any(~np.isfinite(values)) or np.min(values) < -1e-12 or np.max(values) > 1 + 1e-12:
            raise AssertionError(f"invalid {name} probability: {values}")
        sign = -1 if name in ("first_default", "both_default") else 1
        if np.any(sign * np.diff(values) > 1e-10):
            raise AssertionError(f"{name} is not monotone in maturity: {values}")


def qualify():
    """Check the frozen pure-jump table, diffusion bounds, independence and refinements."""
    model, case = load_case()
    dates = case["maturities"]
    config = case["grid"]
    mc = case["monte_carlo"]
    paths = mc["paths"]
    start = perf_counter()
    bench = bridge_benchmark(model, dates, paths, mc["seed"], mc["bridge_intervals_per_year"])
    mc_seconds = perf_counter() - start
    width = bench["monitoring_width"]
    width_limit = np.array(width["estimate"]) + mc["standard_errors"] * np.array(width["se"])
    if np.any(width_limit > mc["max_monitoring_width"]):
        raise AssertionError(f"continuous-monitoring bracket too wide: {width_limit}")
    exact = poisson_gamma(model, dates)
    table = _observables(dict(zip(("x", "y", "joint"), exact.T, strict=True)))
    frozen = case["pure_jump"]
    for key, target in (
        ("joint", "joint_survival"),
        ("first_default", "first_default"),
        ("both_default", "both_default"),
    ):
        np.testing.assert_allclose(table[key], frozen[target], rtol=0, atol=5.1e-9)
    np.testing.assert_allclose(exact[-1, :2], frozen["one_year_marginals"], rtol=0, atol=5.1e-9)
    # Hold the other two controls fixed for each sensitivity.
    prices(model, [0.25], cells_per_distance=4, upper_distances=2, steps_per_year=4)
    runs = {}
    for dynamics, scenario in (("pure_jump", model.without_diffusion()), ("diffusion", model)):
        results = {}
        for name, override in (
            ("coarser_space", {"cells_per_distance": config["cells_per_distance"] // 2}),
            ("coarser_time", {"steps_per_year": config["steps_per_year"] // 2}),
            ("narrower_domain", {"upper_distances": config["upper_distances"] - 1}),
            ("refined", {}),
        ):
            settings = {**config, **override}
            start = perf_counter()
            q = prices(scenario, dates, **settings)
            elapsed = perf_counter() - start
            _valid(q)
            results[name] = dict(
                grid=settings,
                seconds=elapsed,
                probabilities={k: v.tolist() for k, v in _observables(q).items()},
            )
        fine = results["refined"]["probabilities"]
        if dynamics == "pure_jump":
            for key, value in table.items():
                np.testing.assert_allclose(
                    fine[key], value, rtol=0, atol=frozen["absolute_tolerance"]
                )
        else:
            for key in ("x", "y", "joint"):
                lo = bench["joint_lower" if key == "joint" else key]
                hi = bench["joint_upper" if key == "joint" else key]
                if np.any(
                    fine[key]
                    < np.array(lo["estimate"]) - mc["standard_errors"] * np.array(lo["se"])
                ) or np.any(
                    fine[key]
                    > np.array(hi["estimate"]) + mc["standard_errors"] * np.array(hi["se"])
                ):
                    raise AssertionError(
                        f"{key} outside bridge bounds with sampling uncertainty: "
                        f"{fine[key]} vs {lo}, {hi}"
                    )
        reference = (
            exact[:, 2]
            if dynamics == "pure_jump"
            else (np.array(bench["joint_upper"]["estimate"]) + bench["joint_lower"]["estimate"]) / 2
        )
        for row in results.values():
            row["joint_minus_reference"] = (row["probabilities"]["joint"] - reference).tolist()
            row["joint_change_from_refined"] = (
                np.array(row["probabilities"]["joint"]) - fine["joint"]
            ).tolist()
            np.testing.assert_allclose(
                row["joint_change_from_refined"],
                0,
                rtol=0,
                atol=case["refinement_absolute_tolerance"],
            )
        runs[dynamics] = results
    independent = prices(replace(model, common_intensity=0.0, correlation=0.0), dates, **config)
    independence_gap = independent["joint"] - independent["x"] * independent["y"]
    np.testing.assert_allclose(independence_gap, 0, atol=1e-5, rtol=0)
    first = prices(model, dates, **config, payoff=Payoff.FIRST_DEFAULT)["joint"]
    np.testing.assert_allclose(
        first, runs["diffusion"]["refined"]["probabilities"]["first_default"], atol=2e-14, rtol=0
    )
    return dict(
        maturities=dates,
        runs=runs,
        exact_pure_jump={key: values.tolist() for key, values in table.items()},
        bridge=bench,
        mc_paths=paths,
        mc_seconds=mc_seconds,
        mc_seed=mc["seed"],
        bridge_intervals_per_year=mc["bridge_intervals_per_year"],
        independence_joint_minus_product=independence_gap.tolist(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional full qualification JSON.")
    args = parser.parse_args()
    result = qualify()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print("years   pure joint   exact pure   diffusion joint   bridge bounds       max bound SE")
    for j, t in enumerate(result["maturities"]):
        pure, diffusion = (
            result["runs"][k]["refined"]["probabilities"]["joint"][j]
            for k in ("pure_jump", "diffusion")
        )
        lo, hi = (result["bridge"][k] for k in ("joint_lower", "joint_upper"))
        print(
            f"{t:5.2f}  {pure:.8f}   {result['exact_pure_jump']['joint'][j]:.8f}"
            f"   {diffusion:.8f}"
            f"    [{lo['estimate'][j]:.8f}, {hi['estimate'][j]:.8f}]"
            f"   {max(lo['se'][j], hi['se'][j]):.2g}"
        )
