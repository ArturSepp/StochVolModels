"""Timings quoted on docs/numerical_accuracy_and_performance.md.

The page reports one dated run of ``timings`` in a fresh process on a recorded machine. Times depend
on the machine, the library versions, the load and on what the process has already compiled, so
the case asserts only that each operation completes. Run the file to print a new set of timings.
"""
import platform
from enum import Enum
from time import perf_counter

import numpy as np

import stochvolmodels as svm
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.utils.funcs import set_seed

PARAMS = svm.LogSvParams(sigma0=1.0, theta=1.0, kappa1=5.0, kappa2=5.0, beta=0.2, volvol=2.0)


def timed(function) -> float:
    """Wall-clock seconds of one call."""
    start = perf_counter()
    function()
    return perf_counter() - start


def timings() -> dict:
    """Seconds for the first (compiling) and a warm transform price, a chain, and a simulation."""
    pricer = svm.LogSVPricer()
    btc = get_btc_test_chain_data()

    def vanilla():
        pricer.price_vanilla(params=PARAMS, ttm=0.25, forward=1.0, strike=1.0, optiontype="C")

    def chain():
        pricer.price_chain(option_chain=btc, params=PARAMS)

    def simulation():
        set_seed(7)
        pricer.model_mc_price_chain(option_chain=btc, params=PARAMS, nb_path=100000, nb_steps=360)

    out = {"first price, compiling": timed(vanilla), "warm price": timed(vanilla),
           "Bitcoin chain, 4 maturities, 49 options": timed(chain)}
    timed(simulation)  # compile the simulator
    out["Monte Carlo, same chain, 100,000 paths, daily steps"] = timed(simulation)
    return out


class Locals(Enum):
    """Cases of this script."""
    TIMINGS = 1


def run_local(local: Locals) -> None:
    """Run one case and assert what does not depend on the machine or the process."""
    if local == Locals.TIMINGS:
        result = timings()
        print(platform.platform(), platform.processor(), platform.python_version(),
              svm.__version__, np.__version__)
        for name, seconds in result.items():
            print(f"{name}: {seconds:.2f} s")
        assert all(np.isfinite(seconds) and seconds > 0.0 for seconds in result.values())


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
