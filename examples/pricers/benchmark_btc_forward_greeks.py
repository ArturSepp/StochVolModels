"""Offline Bitcoin benchmark for forward delta and gamma on a prepared LogSV grid.

Run from a source checkout with python -m examples.pricers.benchmark_btc_forward_greeks.
The optional --output-dir writes per-quote bumps and the individual timing observations.
The price-only reference uses the same LogSV expansion with independent GL24 quadrature.
"""
import argparse
import json
import os
import platform
from dataclasses import replace
from enum import Enum
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

# Apply the timing convention in the standalone process, before loading a BLAS library.
if __name__ == "__main__":
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[variable] = "1"

import numba
import numpy as np
import pandas as pd
import scipy
from numba.core import event
from scipy.special import roots_legendre

from stochvolmodels import LogSVPricer, LogSvParams, OptionChain
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.models.logsv import affine_expansion as afe


# Full-precision result of the controlled, vega-weighted PARAMS5 IV fit in docs/calibration.md.
FITTED = LogSvParams(sigma0=0.847611699886395, theta=1.0126227894969513,
                     kappa1=3.7476128250437672, kappa2=3.700897179003347,
                     beta=0.16422953116238242, volvol=2.0363136644765523)
PRICE_ATOL = 1e-6
GREEK_TOLERANCES = {"delta_atol": 1e-8, "gamma_atol": 1e-12}
BUMP_FRACTIONS = (2e-4, 1e-4, 5e-5)


def _timed(function, *args, **kwargs):
    start = perf_counter()
    result = function(*args, **kwargs)
    return perf_counter() - start, result


def _central(centre, up, down, h):
    return (up - down) / (2 * h), ((up - centre) + (down - centre)) / h**2


def _reference(chain, cutoff, subdivisions):
    """Return finite premium changes and outright premiums on one independent price grid."""
    edges = np.r_[0., 2.**np.arange(-1, int(np.log2(cutoff)) + 1)]
    left = np.linspace(edges[:-1], edges[1:], subdivisions, endpoint=False).T.ravel()
    half = 0.5 * np.diff(np.r_[left, edges[-1]])
    nodes, weights = roots_legendre(24)
    u = ((left + half)[:, None] + half[:, None] * nodes).ravel()
    weights = (half[:, None] * weights).ravel() / (np.pi * (u*u + 0.25))
    phi = -0.5 + 1j*u
    zero = np.zeros_like(phi)
    premiums = {shift: [] for shift in (0., *BUMP_FRACTIONS, *(-h for h in BUMP_FRACTIONS))}
    changes = {shift: [] for shift in premiums}
    state = None
    for i, dt in enumerate(np.diff(np.r_[0., chain.ttms])):
        state, log_m = afe.compute_logsv_a_mgf_grid(
            ttm=dt, phi_grid=phi, psi_grid=zero, theta_grid=zero, a_t0=state, **FITTED.to_dict())
        strikes, calls = chain.strikes_ttms[i], chain.optiontypes_ttms[i] == "C"
        forward, discount = chain.forwards[i], chain.discfactors[i]
        base = np.exp(log_m - np.log(forward / strikes)[:, None] * phi)
        for shift in premiums:
            bumped = forward * (1 + shift)
            integral = np.exp(log_m - np.log(bumped / strikes)[:, None] * phi).real @ weights
            premiums[shift].append(discount * (np.where(calls, bumped, strikes) - strikes * integral))
            # The finite change of the strike phase keeps the forward bump intact near zero.
            change = (base * np.expm1(-phi * np.log1p(shift))).real @ weights
            changes[shift].append(discount * (np.where(calls, forward * shift, 0.) - strikes * change))
    return ({shift: np.concatenate(values) for shift, values in changes.items()},
            {shift: np.concatenate(values) for shift, values in premiums.items()}, u.size)


def _warm_up(pricer, chain):
    """Compile both query paths and conversion of the returned ragged arrays."""
    warm = OptionChain(ttms=chain.ttms[:2], forwards=chain.forwards[:2], ids=None,
                       strikes_ttms=(np.array([66000.]), np.array([70000.])),
                       optiontypes_ttms=(np.array(["P"]), np.array(["C"])),
                       discfactors=chain.discfactors[:2])
    grid = pricer.prepare_fourier(warm, FITTED).refined_for(warm, **GREEK_TOLERANCES)
    for values in grid.forward_greeks(warm, **GREEK_TOLERANCES):
        np.concatenate(values)
    np.concatenate(grid.price_chain(warm))


def _measure(chain, forwards):
    pricer = LogSVPricer()
    compile_times = []
    with event.install_timer("numba:compile", compile_times.append):
        warmup, _ = _timed(_warm_up, pricer, chain)

    solves = 0
    solver = afe.compute_logsv_a_mgf_grid

    def counted(**kwargs):
        nonlocal solves
        solves += len(kwargs["phi_grid"])
        return solver(**kwargs)

    step = BUMP_FRACTIONS[-1]
    scenarios = [replace(chain, forwards=chain.forwards * (1 + shift)) for shift in (0., step, -step)]
    setup, analytic_times, bump_times, timed_compiles = [], [], [], []
    with patch.object(afe, "compute_logsv_a_mgf_grid", counted), event.install_timer(
            "numba:compile", timed_compiles.append):
        for _ in range(3):
            start = solves
            prepare, grid = _timed(pricer.prepare_fourier, chain, FITTED, fourier={"price_atol": PRICE_ATOL})
            prepared_work = solves - start
            refine, risk = _timed(grid.refined_for, chain, **GREEK_TOLERANCES)
            setup.append(dict(prepare_s=prepare, refine_s=refine, prepare_solves=prepared_work,
                              refine_solves=solves-start-prepared_work, price_nodes=grid._plan.nodes.size,
                              risk_nodes=risk._plan.nodes.size, price_cutoff=grid.diagnostics.cutoff,
                              risk_cutoff=risk.diagnostics.cutoff))

        def central_on_grid():
            prices = [np.concatenate(risk.price_chain(quotes)) for quotes in scenarios]
            return _central(*prices, step * forwards)

        start = solves
        risk.forward_greeks(chain, **GREEK_TOLERANCES)
        central_on_grid()
        for i in range(51):
            if i % 2 == 0:
                bump_time, _ = _timed(central_on_grid)
            analytic_time, greeks = _timed(risk.forward_greeks, chain, **GREEK_TOLERANCES)
            if i % 2:
                bump_time, _ = _timed(central_on_grid)
            analytic_times.append(analytic_time)
            bump_times.append(bump_time)
    assert solves == start, "a warm query launched a coefficient solve"
    assert not sum(timed_compiles), "Numba compiled during a measured phase"
    timings = dict(warmup_s=warmup, numba_compile_s=sum(compile_times),
                   timed_numba_compile_s=sum(timed_compiles), setup=setup,
                   warm_greeks_s=analytic_times, warm_bumps_s=bump_times,
                   warm_query_coefficient_solves=solves-start)
    return [np.concatenate(values) for values in greeks], timings


class Locals(Enum):
    """Bundled benchmark cases."""
    BITCOIN = 1


def run_local(local: Locals, output_dir: Path | None = None) -> None:
    """Run the full benchmark, check both Greeks, and optionally export the measurements."""
    if local != Locals.BITCOIN:
        raise ValueError(f"unknown benchmark case {local}")
    chain = get_btc_test_chain_data()
    sizes = [len(strikes) for strikes in chain.strikes_ttms]
    assert sizes == [12, 13, 15, 9]
    forwards = np.repeat(chain.forwards, sizes)
    analytic, timings = _measure(chain, forwards)
    quotes = pd.DataFrame(dict(expiry=np.repeat(chain.ids, sizes), days=np.repeat(365 * chain.ttms, sizes),
                               forward=forwards, strike=np.concatenate(chain.strikes_ttms),
                               option=np.concatenate(chain.optiontypes_ttms), delta=analytic[0], gamma=analytic[1]))
    cutoff = 2 ** int(np.ceil(np.log2(max(256., timings["setup"][-1]["risk_cutoff"]))))
    coarse, _, coarse_nodes = _reference(chain, cutoff, 4)
    fine, premiums, fine_nodes = _reference(chain, 2 * cutoff, 8)
    errors = []
    for step in BUMP_FRACTIONS:
        h = step * forwards
        reference = _central(fine[0.], fine[step], fine[-step], h)
        previous = _central(coarse[0.], coarse[step], coarse[-step], h)
        outright = _central(premiums[0.], premiums[step], premiums[-step], h)
        for i, name in enumerate(("delta", "gamma")):
            gap = analytic[i] - reference[i]
            tolerance = GREEK_TOLERANCES[f"{name}_atol"]
            # The second-order central difference has a truncation error proportional to h squared.
            np.testing.assert_allclose(analytic[i], reference[i], rtol=0, equal_nan=False,
                                       atol=tolerance * (step / BUMP_FRACTIONS[-1])**2)
            np.testing.assert_allclose(reference[i], previous[i], rtol=0, atol=tolerance, equal_nan=False)
            worst = quotes.loc[np.argmax(np.abs(gap))]
            errors.append(dict(bump_bp=step * 10000, greek=name, max_abs=np.max(np.abs(gap)),
                               rmse=np.sqrt(np.mean(gap**2)), worst_expiry=worst.expiry,
                               worst_strike=worst.strike, worst_option=worst.option,
                               raw_max_abs=np.max(np.abs(analytic[i] - outright[i])),
                               reference_grid_change=np.max(np.abs(reference[i] - previous[i]))))
            quotes[f"{name}_bump_{step:g}"] = reference[i]
            quotes[f"{name}_error_{step:g}"] = gap
            quotes[f"{name}_raw_bump_{step:g}"] = outright[i]

    summary = pd.DataFrame(errors)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f'Warm-up {timings["warmup_s"]:.3f} s including {timings["numba_compile_s"]:.3f} s Numba compilation')
    for phase in ("prepare", "refine"):
        seconds = [sample[f"{phase}_s"] for sample in timings["setup"]]
        work = [sample[f"{phase}_solves"] for sample in timings["setup"]]
        print(f"{phase}: median {np.median(seconds):.6f} s over {len(work)} runs, coefficient solves {work}")
    for label in ("warm_greeks_s", "warm_bumps_s"):
        print(f"{label}: median {np.median(timings[label])*1000:.3f} ms over {len(timings[label])} runs")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        quotes.to_csv(output_dir / "btc_forward_greeks_quotes.csv", index=False)
        report = dict(chain="BTC 2021-10-21", quotes_per_expiry=sizes, params=FITTED.to_dict(),
                      measure="money-market", expansion_order=2, price_atol=PRICE_ATOL, **GREEK_TOLERANCES,
                      platform=platform.platform(), cpu=platform.processor(), python=platform.python_version(),
                      numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
                      openblas_threads=os.environ.get("OPENBLAS_NUM_THREADS"), **timings,
                      reference_cutoffs=[cutoff, 2*cutoff], reference_nodes=[coarse_nodes, fine_nodes],
                      errors=summary.to_dict("records"))
        (output_dir / "btc_forward_greeks_results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, help="Directory for the per-quote CSV and timing/error JSON")
    run_local(Locals.BITCOIN, output_dir=parser.parse_args().output_dir)
