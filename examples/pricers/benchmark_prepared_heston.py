"""Benchmark ten-year Heston premiums, forward delta and forward gamma on a prepared grid.

Run python -m examples.pricers.benchmark_prepared_heston from a source checkout.
Each timed setup builds a new handle. Warm queries recompute all premiums. The shifted
forward also exercises the fixed-grid checks for quotes outside the preparation set.
Pass --v0 to benchmark initial-variance scenarios and d(premium)/d(v0).
Greek accuracy uses central bumps of fresh high-accuracy premiums. Timed warm bumps
use a separate grid certified at all forwards or variances needed for the smallest bump.
"""

import argparse
import json
import os
import platform
from dataclasses import replace
from enum import Enum
from time import perf_counter_ns
from unittest.mock import patch

if __name__ == "__main__":
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[variable] = "1"

import numba
import numpy as np
from stochvolmodels import HestonParams, HestonPricer, OptionChain
from stochvolmodels.models.heston import HestonModel


class Locals(Enum):
    TEN_YEAR = 1
    INITIAL_VARIANCE = 2


GREEK_TOLERANCES = dict(delta_atol=1e-9, gamma_atol=1e-10)
BUMP_PRICE_ATOL = 1e-11
FORWARD_BUMPS = (0.02, 0.01, 0.005)
VARIANCES = (0.01, 0.04, 0.09)
VARIANCE_BUMPS = (1e-4, 5e-5, 2.5e-5, 1.25e-5)
VARIANCE_ATOL = 1e-8


def _chain(strikes):
    return OptionChain.slice_to_chain(10.0, 100.0, strikes, np.full(strikes.size, "C"), 1.0)


def _timings(operations, rounds):
    """Interleave warmed operations and report milliseconds per complete invocation."""
    rng = np.random.default_rng(6833)
    elapsed = {name: [] for name in operations}
    for action in operations.values():
        action()
    for _ in range(rounds):
        for name in rng.permutation(list(operations)):
            start = perf_counter_ns()
            operations[name]()
            elapsed[name].append((perf_counter_ns() - start) / 1e6)
    return {
        name: dict(zip(("p10", "median", "p90"), np.percentile(times, (10, 50, 90)), strict=True))
        for name, times in elapsed.items()
    }


def _central(centre, up, down, h):
    return (up - down) / (2 * h), ((up - centre) + (down - centre)) / h**2


def _greek_case(pricer, params, quotes, grid, rounds):
    """Compare the analytic pair to price-only bumps at fixed strikes and dynamics."""

    def refine():
        return grid.refined_for(quotes, **GREEK_TOLERANCES)

    risk = refine()

    def greeks():
        delta, gamma = risk.forward_greeks(quotes, **GREEK_TOLERANCES)
        return delta[0], gamma[0]

    def bumped(h):
        return replace(quotes, forwards=quotes.forwards + h)

    def fresh(chain):
        return pricer.price_chain(chain, params, fourier={"price_atol": BUMP_PRICE_ATOL})[0]

    strikes = quotes.strikes_ttms[0]
    wing = int(np.flatnonzero(strikes == 140.0)[0])
    centre = fresh(quotes)
    references = {
        h: _central(centre, fresh(bumped(h)), fresh(bumped(-h)), h) for h in FORWARD_BUMPS
    }
    with patch.object(
        HestonModel, "_affine_coefficients", side_effect=AssertionError("warm transform")
    ):
        delta, gamma = greeks()
    errors = {}
    for h, (bump_delta, bump_gamma) in references.items():
        delta_error = np.abs(delta - bump_delta)
        gamma_error = np.abs(gamma - bump_gamma)
        errors[h] = dict(
            max_delta=float(delta_error.max()),
            max_gamma=float(gamma_error.max()),
            rms_delta=float(np.sqrt(np.mean(delta_error**2))),
            rms_gamma=float(np.sqrt(np.mean(gamma_error**2))),
            worst_delta_strike=float(strikes[delta_error.argmax()]),
            worst_gamma_strike=float(strikes[gamma_error.argmax()]),
        )

    h = FORWARD_BUMPS[-1]
    up, down = bumped(h), bumped(-h)

    def prepare_bumps():
        price_grid = pricer.prepare_fourier(quotes, params, fourier={"price_atol": BUMP_PRICE_ATOL})
        return price_grid.refined_for(up).refined_for(down)

    bump_grid = prepare_bumps()

    def warm_bumps():
        return _central(
            bump_grid.price_chain(quotes)[0],
            bump_grid.price_chain(up)[0],
            bump_grid.price_chain(down)[0],
            h,
        )

    with patch.object(
        HestonModel, "_affine_coefficients", side_effect=AssertionError("warm transform")
    ):
        warm_delta, warm_gamma = warm_bumps()
    reference_delta, reference_gamma = references[h]
    np.testing.assert_allclose(delta, reference_delta, atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(gamma, reference_gamma, atol=2e-8, rtol=0.0)
    np.testing.assert_allclose(warm_delta, reference_delta, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(warm_gamma, reference_gamma, atol=2e-8, rtol=0.0)
    return dict(
        wing_delta=float(delta[wing]),
        wing_gamma=float(gamma[wing]),
        central_errors=errors,
        refine_transforms=risk.diagnostics.evaluations - grid.diagnostics.evaluations,
        bump_setup_transforms=bump_grid.diagnostics.evaluations,
        warm_transforms=0,
        milliseconds=_timings(
            {
                "refine": refine,
                "warm_greeks": greeks,
                "prepare_bumps": prepare_bumps,
                "warm_bumps": warm_bumps,
            },
            rounds,
        ),
    )


def _case(pricer, params, quotes, rounds):
    changed = replace(quotes, forwards=quotes.forwards * 1.0001)

    def prepare():
        return pricer.prepare_fourier(quotes, params, fourier={"price_atol": 1e-6})

    def fresh(chain):
        return pricer.price_chain(chain, params, fourier={"price_atol": 1e-6})[0]

    grid = prepare()
    expected, changed_expected = fresh(quotes), fresh(changed)
    # Any model call here would invalidate the zero-work comparison.
    with patch.object(
        HestonModel, "_affine_coefficients", side_effect=AssertionError("warm transform")
    ):
        base = grid.price_chain(quotes)[0]
        moved = grid.price_chain(changed)[0]
    np.testing.assert_allclose(base, expected, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(moved, changed_expected, atol=1e-6, rtol=0.0)
    wing = int(np.flatnonzero(quotes.strikes_ttms[0] == 140.0)[0])
    np.testing.assert_allclose(base[wing], 0.295774435798, atol=1e-6, rtol=0.0)
    times = _timings(
        {
            "fresh": lambda: fresh(quotes),
            "prepare": prepare,
            "warm": lambda: grid.price_chain(quotes)[0],
            "warm_changed_forward": lambda: grid.price_chain(changed)[0],
        },
        rounds,
    )
    return dict(
        strikes=len(base),
        wing=float(base[wing]),
        max_error_fresh=float(max(abs(base - expected))),
        max_error_changed=float(max(abs(moved - changed_expected))),
        preparation_transforms=grid.diagnostics.evaluations,
        warm_transforms=0,
        retained_nodes=grid.diagnostics.panels * 15,
        cutoff=grid.diagnostics.cutoff,
        milliseconds=times,
        forward_greeks=_greek_case(pricer, params, quotes, grid, rounds),
    )


def _variance_case(pricer, params, quotes, rounds):
    """Measure a complete sweep over three initial variances at fixed Heston dynamics."""
    price_controls = {"price_atol": 1e-6}
    reference_controls = {"price_atol": BUMP_PRICE_ATOL}

    def prepare():
        return pricer.prepare_fourier(quotes, params, fourier=price_controls)

    def fresh(state, controls):
        return pricer.price_chain(quotes, replace(params, v0=state), fourier=controls)[0]

    seed = prepare()

    def refine():
        grid = seed
        for state in VARIANCES:
            grid = grid.refined_for(quotes, v0=state, sensitivity_atol=VARIANCE_ATOL)
        return grid

    risk = refine()

    def prices():
        return [risk.price_chain(quotes, v0=v)[0] for v in VARIANCES]

    def sensitivities():
        return [risk.v0_sensitivity(quotes, v0=v, atol=VARIANCE_ATOL)[0] for v in VARIANCES]

    h = VARIANCE_BUMPS[-1]

    def prepare_bumps():
        grid = pricer.prepare_fourier(quotes, params, fourier=reference_controls)
        for state in VARIANCES:
            for bumped in (state + h, state - h):
                grid = grid.refined_for(quotes, v0=bumped)
        return grid

    bump_grid = prepare_bumps()

    def warm_bumps():
        return [
            (
                bump_grid.price_chain(quotes, v0=state + h)[0]
                - bump_grid.price_chain(quotes, v0=state - h)[0]
            )
            / (2 * h)
            for state in VARIANCES
        ]

    with patch.object(
        HestonModel, "_affine_coefficients", side_effect=AssertionError("warm solve")
    ):
        scenario_prices, analytic, bumped_derivatives = prices(), sensitivities(), warm_bumps()

    strikes = quotes.strikes_ttms[0]
    wing = int(np.flatnonzero(strikes == 140.0)[0])
    result = {}
    for i, state in enumerate(VARIANCES):
        reference = fresh(state, reference_controls)
        price_errors = np.abs(scenario_prices[i] - reference)
        np.testing.assert_allclose(
            scenario_prices[i], reference, rtol=0.0, atol=price_controls["price_atol"]
        )
        central_by_step = {
            step: (
                fresh(state + step, reference_controls) - fresh(state - step, reference_controls)
            )
            / (2 * step)
            for step in VARIANCE_BUMPS
        }
        bumps = {}
        for step, central in central_by_step.items():
            errors = np.abs(analytic[i] - central)
            bumps[step] = dict(
                max_error=float(errors.max()),
                rms_error=float(np.sqrt(np.mean(errors**2))),
                worst_strike=float(strikes[errors.argmax()]),
                wing_error=float(errors[wing]),
            )
        np.testing.assert_allclose(analytic[i], central_by_step[h], atol=2e-7, rtol=0.0)
        np.testing.assert_allclose(bumped_derivatives[i], central_by_step[h], atol=2e-8, rtol=0.0)
        result[state] = dict(
            wing_price=float(scenario_prices[i][wing]),
            wing_sensitivity=float(analytic[i][wing]),
            max_price_error=float(price_errors.max()),
            worst_price_strike=float(strikes[price_errors.argmax()]),
            central_bumps=bumps,
        )

    timings = _timings(
        {
            "prepare": prepare,
            "certify_states_and_sensitivities": refine,
            "warm_scenario_prices": prices,
            "fresh_scenario_prices": lambda: [fresh(v, price_controls) for v in VARIANCES],
            "warm_sensitivities": sensitivities,
            "warm_central_bumps": warm_bumps,
        },
        rounds,
    )
    return dict(
        strikes=len(strikes),
        states=result,
        preparation_transforms=seed.diagnostics.evaluations,
        refinement_transforms=risk.diagnostics.evaluations - seed.diagnostics.evaluations,
        bump_setup_transforms=bump_grid.diagnostics.evaluations,
        warm_transforms=0,
        milliseconds=timings,
    )


def run_local(local=Locals.TEN_YEAR):
    """Return the benchmark settings, accuracy checks and warm timing distributions."""
    pricer = HestonPricer()
    params = HestonParams(v0=0.04, theta=0.04, kappa=0.5, rho=-0.9, volvol=1.0)
    wing, dense = _chain(np.array([140.0])), _chain(np.linspace(50.0, 150.0, 1001))
    environment = dict(
        python=platform.python_version(),
        os=platform.platform(),
        processor=platform.processor(),
        numpy=np.__version__,
        numba=numba.__version__,
    )
    if local == Locals.INITIAL_VARIANCE:
        rounds = 21
        return dict(
            **environment,
            rounds=rounds,
            seed_v0=params.v0,
            states=VARIANCES,
            variance_bumps=VARIANCE_BUMPS,
            sensitivity_atol=VARIANCE_ATOL,
            price_atol=1e-6,
            reference_price_atol=BUMP_PRICE_ATOL,
            wing=_variance_case(pricer, params, wing, rounds),
            dense=_variance_case(pricer, params, dense, rounds),
        )
    if local != Locals.TEN_YEAR:
        raise ValueError(f"unknown benchmark {local}")
    rounds = 31
    result = dict(
        **environment,
        rounds=rounds,
        greek_tolerances=GREEK_TOLERANCES,
        bump_price_atol=BUMP_PRICE_ATOL,
        forward_bumps=FORWARD_BUMPS,
        wing=_case(pricer, params, wing, rounds),
        dense=_case(pricer, params, dense, rounds),
    )
    seed = pricer.prepare_fourier(wing, params)
    extended = seed.refined_for(dense)
    reference = pricer.price_chain(dense, params, fourier={})[0]
    error = float(max(abs(extended.price_chain(dense)[0] - reference)))
    assert error <= 1e-6
    result["wing_to_dense"] = dict(
        additional_transforms=extended.diagnostics.evaluations - seed.diagnostics.evaluations,
        max_error_fresh=error,
        milliseconds=_timings(
            {
                "refine": lambda: seed.refined_for(dense),
                "warm": lambda: extended.price_chain(dense)[0],
            },
            rounds,
        ),
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v0", action="store_true", help="run the initial-variance benchmark")
    print(
        json.dumps(
            run_local(Locals.INITIAL_VARIANCE if parser.parse_args().v0 else Locals.TEN_YEAR),
            indent=2,
        )
    )
