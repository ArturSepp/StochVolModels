"""Timings and the LogSV pricing comparison in docs/numerical_accuracy_and_performance.md.

The page reports one dated run of ``timings`` in a fresh process on a recorded machine. Times depend
on the machine, the library versions, the load and on what the process has already compiled, so
the case asserts only that each operation completes. Run the file to print a new set of timings.
Pass --fourier-report with the paper verification JSON to compare the public pricer with its
state-grid PDE and independent simulation on the 12-day SPY calls.
"""
import argparse
import json
import platform
from enum import Enum
from pathlib import Path
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


def compare_spy_calls(report: Path) -> None:
    """Print unit-forward premiums alongside the full-state references for 12-day SPY."""
    from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification import fourier

    cases = json.loads(report.read_text(encoding="utf-8"))["checks"].get("pricing", [])
    case = next((row for row in cases if row["case"] == "published_spy_12d"), None)
    if case is None:
        raise ValueError("The 12-day SPY prices require the full paper verification report.")
    assert case["reference_passed"]
    params = svm.LogSvParams(**case["parameters"])
    strikes = np.asarray(case["strikes_over_forward"])
    calls = dict(params=params, ttm=case["maturity"], forward=1.0, strikes=strikes,
                 optiontypes=np.full(strikes.size, "C"), discfactor=1.0)
    pricer = svm.LogSVPricer()
    legacy, _ = pricer.price_slice(**calls, fourier="legacy")
    controlled, _ = pricer.price_slice(**calls, fourier={"price_atol": 1e-6})
    phi, weights = fourier.gaussian_grid(order=48, cutoff=2048.)
    zeros = np.zeros_like(phi)
    _, log_mgf = svm.compute_logsv_a_mgf_grid(ttm=case["maturity"], phi_grid=phi,
                                              psi_grid=zeros, theta_grid=zeros, **params.to_dict())
    reference = fourier.calls_from_transform(np.exp(log_mgf), phi, weights, case["maturity"],
                                             params.sigma0, strikes)
    np.testing.assert_allclose(controlled, reference, rtol=0, atol=1e-6)
    # The observed full-state gap on these three calls is below 0.01 bp of forward.
    np.testing.assert_allclose(controlled, case["pde_prices"], rtol=0, atol=1e-6)
    print("K/F    legacy       controlled   PDE          Monte Carlo   MC SE (bp of forward)")
    for k, old, price, pde, mc, se in zip(
        strikes, legacy, controlled, case["pde_prices"], case["mc_prices"],
        case["mc_scramble_se_bp"], strict=True,
    ):
        print(f"{k:.2f}   {old:.9f}  {price:.9f}  {pde:.9f}  {mc:.9f}   {se:.5f}")
    gap = np.max(np.abs(controlled - reference))
    print(f"Maximum premium difference from same-transform Gauss–Legendre inversion = {gap:.3g}")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fourier-report", type=Path, help="JSON from the full IJTAF verification")
    args = parser.parse_args()
    if args.fourier_report is not None:
        compare_spy_calls(args.fourier_report)
    else:
        for case in Locals:
            run_local(local=case)
