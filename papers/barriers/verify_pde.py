"""Verify the conditional 1D barrier adapter against the active paper replication."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

from . import green_solutions as paper
from .pde_adapter import BarrierPde1D


def max_error(a, b):
    """Largest absolute difference on the comparison grid."""
    return float(np.max(np.abs(a - b)))


def check_partition():
    """Reproduce the roadmap example on its original finite domain and time partition."""
    maturity, mt = 0.06, 2
    times = np.linspace(0.0, maturity, 11)
    drift = 0.02 * np.cos(30.0 * times)
    variance = 0.04 * (1.0 + np.sin(50.0 * times))
    x = np.linspace(-0.05, 0.20, 101)
    adapter = BarrierPde1D(x)
    green = adapter.green(maturity, mt, drift, variance)
    full = adapter.survival(maturity, mt, drift, variance)[:, 0]
    historical = adapter.survival(maturity, mt, drift, variance, partition="historical")[:, 0]
    reference_g = paper.compute_pde_green(maturity, mt, x, drift, variance)
    reference_s = paper.compute_pde_survival_prob(maturity, mt, x, drift, variance)
    np.testing.assert_allclose(green.density[:, 0], reference_g, rtol=0, atol=2e-10)
    np.testing.assert_allclose(historical, reference_s, rtol=0, atol=1e-12)
    j = int(np.argmin(abs(x)))
    # Upper truncation has value one for backward down-barrier survival.
    np.testing.assert_allclose(full[j], green.surviving_mass + green.upper_exit, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        green.lower_exit + green.upper_exit + green.surviving_mass, 1, atol=1e-12, rtol=0
    )
    worst = int(np.argmax(np.abs(historical - full)))
    return dict(
        maturity=maturity,
        nt=len(times) - 1,
        substeps=mt + 1,
        omitted_time=times[1],
        omitted_integrated_variance=times[1] * variance[1],
        omitted_drift=times[1] * drift[1],
        survivor=float(green.surviving_mass[0]),
        lower_exit=float(green.lower_exit[0]),
        upper_exit=float(green.upper_exit[0]),
        max_green_error=max_error(green.density[:, 0], reference_g),
        max_historical_error=max_error(historical, reference_s),
        max_partition_gap=max_error(historical, full),
        worst_x=float(x[worst]),
        historical_at_zero=float(historical[j]),
        full_at_zero=float(full[j]),
        x=x.tolist(),
        full=full.tolist(),
        historical=historical.tolist(),
    )


def _reflection(x, barrier, drift, variance, maturity):
    """One lower barrier on a half-line, starting at zero for the Green density."""
    sd = np.sqrt(variance * maturity)
    shift = drift * maturity

    def gaussian(z):
        return np.exp(-0.5 * (z / sd) ** 2) / (np.sqrt(2 * np.pi) * sd)

    green = gaussian(x - shift) - np.exp(2 * drift * barrier / variance) * gaussian(
        x - 2 * barrier - shift
    )
    distance = x - barrier
    survival = ndtr((distance + shift) / sd) - np.exp(-2 * drift * distance / variance) * ndtr(
        (shift - distance) / sd
    )
    return green, survival


def check_independent():
    """Compare to the single-barrier Volterra equations and constant-coefficient images."""
    T, barrier = 0.06, -0.05
    rows = []
    for label in ("constant", "time_varying"):
        for n in (10, 20, 40):
            dates = np.linspace(0.0, T, n + 1)
            drift = np.full(n + 1, 0.02)
            var = np.full(n + 1, 0.04)
            if label == "time_varying":
                drift *= np.cos(30.0 * dates)
                var *= 1.0 + np.sin(50.0 * dates)
            # Resolve the reflection density, with negligible exit at the distant upper cutoff.
            x = np.linspace(barrier, 0.5, 441)
            adapter = BarrierPde1D(x)
            mt = 15
            law = adapter.green(T, mt, drift, var)
            full = adapter.survival(T, mt, drift, var)[:, 0]
            hist = adapter.survival(T, mt, drift, var, partition="historical")[:, 0]
            vol_g = paper.compute_volterra_green(T, mt, barrier, x, drift, var)
            vol_s = paper.compute_volterra_survival_prob(T, mt, barrier, x, drift, var)
            vol_mass = adapter.grid.integrate(vol_g)
            # Discounted normalized calls on the surviving interior law.
            strikes = np.array([0.97, 1.0, 1.05])
            payoffs = np.maximum(np.exp(x[:, None]) - strikes, 0.0)
            discounted_weights = np.exp(-0.03 * T) * adapter.grid.quadrature_weights
            calls = (discounted_weights * law.density[:, 0]) @ payoffs
            vol_calls = (discounted_weights * vol_g) @ payoffs
            np.testing.assert_allclose(calls, vol_calls, rtol=0, atol=1e-5)
            row = dict(
                case=label,
                nt=n,
                substeps=mt + 1,
                upper_exit=float(law.upper_exit[0]),
                max_green_volterra=max_error(law.density[:, 0], vol_g),
                mass_gap_volterra=float(law.surviving_mass[0] - vol_mass),
                strikes=strikes.tolist(),
                call_vs_volterra=max_error(calls, vol_calls),
                full_vs_volterra=max_error(full, vol_s),
                historical_vs_volterra=max_error(hist, vol_s),
                max_partition_gap=max_error(full, hist),
            )
            if label == "constant":
                ref_g, ref_s = _reflection(x, barrier, drift[1], var[1], T)
                _, truncated = _reflection(x, barrier, drift[1], var[1], T - dates[1])
                row.update(
                    full_vs_analytic=max_error(full, ref_s),
                    historical_vs_full_analytic=max_error(hist, ref_s),
                    historical_vs_short_analytic=max_error(hist, truncated),
                    max_green_analytic=max_error(law.density[:, 0], ref_g),
                )
                np.testing.assert_allclose(full, ref_s, rtol=0, atol=3e-5)
                np.testing.assert_allclose(hist, truncated, rtol=0, atol=3e-5)
                np.testing.assert_allclose(law.density[:, 0], ref_g, rtol=0, atol=0.001)
            np.testing.assert_allclose(full, vol_s, rtol=0, atol=4e-4)
            np.testing.assert_allclose(law.density[:, 0], vol_g, rtol=0, atol=0.002)
            np.testing.assert_allclose(law.surviving_mass, vol_mass, rtol=0, atol=3e-5)
            assert row["upper_exit"] < 1e-10
            rows.append(row)
    return rows


def _timed(fn, *args, **kwargs):
    start = perf_counter()
    value = fn(*args, **kwargs)
    return value, perf_counter() - start


def check_path_batches():
    """Compare the three full-size Numba PDE figure batches on the same simulated paths."""
    from . import article_figures as figures
    from .mc_engine import simulate_heston_decomposition_paths
    from .mc_volterra import FunctionType, compute_heston_volterra_barrier

    params = dict(t=1.0, v0=0.25, theta=0.2, kappa=1.0, rho=-0.3, volvol=0.4)
    rows = []
    for kind, paths, nodes, lower, rate, seed in (
        (FunctionType.GREEN, 5000, 200, -0.5, 0.0, 2),
        (FunctionType.SURVIVAL_PROB, 6000, 400, -0.5, 0.03, 3),
        (FunctionType.CALL, 2000, 400, np.log(0.9), 0.03, 3),
    ):
        x = np.linspace(lower, 2.0, nodes)
        strikes = np.log(np.linspace(0.9, 1.5, 13))
        v, integral, drift = simulate_heston_decomposition_paths(
            **params, nb_path=paths, rfrate=rate, seed_id=seed
        )
        var = (1 - params["rho"] ** 2) * v
        adapter = BarrierPde1D(x)
        discount = np.exp(-rate * params["t"])

        def evaluate(engine, v_paths, i_paths, d_paths, partition="historical"):
            return figures._figure_pde(
                engine,
                partition=partition,
                **params,
                x0=0.0,
                rfrate=rate,
                x_grid=x,
                nb_path=d_paths.shape[1],
                mt_=6,
                vt=v_paths,
                it=i_paths,
                mt=d_paths,
                log_strikes=strikes,
                function_type=kind,
            )

        # Match compiled array layouts and exclude simulation and first compilation from timing.
        warm_paths = (v[:, :2].copy(), integral[:, :2].copy(), drift[:, :2].copy())
        for engine in ("paper", "adapter"):
            evaluate(engine, *warm_paths)
        ref, original_s = _timed(evaluate, "paper", v, integral, drift)
        result, adapter_s = _timed(evaluate, "adapter", v, integral, drift)
        np.testing.assert_allclose(result, ref, rtol=0, atol=2e-10)
        row = dict(
            function=kind.name,
            seed=seed,
            paths=paths,
            nodes=nodes,
            intervals=len(v) - 1,
            substeps=7,
            discount=discount,
            paper_seconds=original_s,
            adapter_seconds=adapter_s,
            max_paper_error=max_error(result, ref),
            reference=ref.tolist(),
            adapter=result.tolist(),
            coordinates=(np.exp(strikes) if kind is FunctionType.CALL else x).tolist(),
        )
        if kind is FunctionType.SURVIVAL_PROB:
            full, secs = _timed(evaluate, "adapter", v, integral, drift, "full")
            j = int(np.argmax(np.abs(full - result)))
            row.update(
                full=full.tolist(),
                full_seconds=secs,
                max_discounted_partition_gap=max_error(full, result),
                worst_x=float(x[j]),
                historical_at_worst=float(result[j]),
                full_at_worst=float(full[j]),
                undiscounted_gap=max_error(full, result) / discount,
            )
        else:
            law = adapter.green(params["t"], 6, drift, var)
            mass = law.surviving_mass + law.lower_exit + law.upper_exit
            row.update(
                mean_surviving_mass=float(law.surviving_mass.mean()),
                mean_lower_exit=float(law.lower_exit.mean()),
                mean_upper_exit=float(law.upper_exit.mean()),
                min_lower_exit=float(law.lower_exit.min()),
                min_upper_exit=float(law.upper_exit.min()),
                max_mass_error=max_error(mass, 1.0),
                max_terminal_negative_mass=float(
                    np.max(adapter.grid.quadrature_weights @ np.maximum(-law.density, 0.0))
                ),
            )
            np.testing.assert_allclose(mass, 1.0, rtol=0, atol=1e-11)
            if kind is FunctionType.CALL:
                # Independent payoff projection using trapezoid weights and the mean law.
                payoffs = np.maximum(np.exp(x[:, None]) - np.exp(strikes), 0.0)
                weight = adapter.grid.quadrature_weights * law.density.mean(axis=1)
                np.testing.assert_allclose(discount * (weight @ payoffs), ref, rtol=0, atol=2e-10)
        if kind in (FunctionType.CALL, FunctionType.SURVIVAL_PROB):
            volterra = compute_heston_volterra_barrier(
                **params,
                psi0=lower,
                x0=0.0,
                x_grid=x,
                rfrate=rate,
                mt_=6,
                nb_path=paths,
                vt=v,
                it=integral,
                mt=drift,
                log_strikes=strikes,
                function_type=kind,
            )
            row["volterra"] = volterra.tolist()
            if kind is FunctionType.SURVIVAL_PROB:
                # The historical Volterra approximation can be negative near the barrier.
                worst = int(np.argmax(np.abs(full - volterra)))
                row.update(
                    historical_vs_volterra=max_error(result, volterra),
                    full_vs_volterra=max_error(full, volterra),
                    worst_volterra_x=float(x[worst]),
                    minimum_volterra=float(volterra.min()),
                    volterra_at_max_partition_gap=float(volterra[j]),
                )
                np.testing.assert_allclose(full, volterra, rtol=0, atol=0.02)
            else:
                np.testing.assert_allclose(result, volterra, rtol=0, atol=2e-4)
                row["max_volterra_error"] = max_error(result, volterra)
        rows.append(row)
    return rows


def main():
    """Run the deterministic cases and optionally benchmark the original figure batches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path-batches", action="store_true", help="also run the original 2000–6000 path batches"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(partition=check_partition(), independent=check_independent())
    if args.path_batches:
        report["batches"] = check_path_batches()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
