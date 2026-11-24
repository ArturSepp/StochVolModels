"""Benchmark ten-year Heston premiums on a prepared grid.

Run python -m examples.pricers.benchmark_prepared_heston from a source checkout.
Each timed setup builds a new handle. Warm queries recompute all premiums. The shifted
forward exercises the fixed-grid checks for quotes outside the preparation set.
"""

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
        HestonModel, "log_mgf_grid", side_effect=AssertionError("warm transform")
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
    if local != Locals.TEN_YEAR:
        raise ValueError(f"unknown benchmark {local}")
    rounds = 31
    result = dict(
        **environment,
        rounds=rounds,
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
    print(json.dumps(run_local(), indent=2))
