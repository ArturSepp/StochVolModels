"""Bounded local-expansion experiments against the existing independent LogSV PDE.

Run from the repository root with python -m
papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.expansion_diagnostics_run
--output agents/<run>/expansion_diagnostics.json. This reports approximation failures;
it does not fit or tune parameters to reference prices, and does not update fixtures.
"""
import argparse
from dataclasses import asdict
from enum import Enum
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numpy.polynomial.polynomial import polyadd, polyder, polymul, polyval

from stochvolmodels import LogSvParams
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.pricers.logsv import affine_expansion as canonical
from . import expansion_diagnostics as experimental
from .expansion_diagnostics import ExpansionConfig
from .logvol_pde import PdeGrid, call_prices as pde_calls


def algebra_checks():
    """Check generic projection, canonical degree four, translations, and exact Black limit."""
    rng = np.random.default_rng(20261030)
    maximum_generator_error, maximum_translation_error, maximum_canonical_error = 0., 0., 0.
    p = LogSvParams(sigma0=.27, theta=.31, kappa1=3., kappa2=4., beta=-.7, volvol=.8)
    phi = np.array([-.5+.3j, -.5+2j])
    for degree in [2, 4, 6, 8]:
        for center in [.2, p.theta, .6]:
            rate = -.02
            a = rng.normal(size=(len(phi), degree+1))+1j*rng.normal(size=(len(phi), degree+1))
            actual = experimental.polynomial_rhs(a.ravel(), phi, degree, center, rate,
                       p.theta, p.kappa1, p.kappa2, p.beta, p.volvol).reshape(a.shape)
            for q, z in enumerate(phi):
                sigma2 = np.array([1., 2., 1.])
                drift = np.array([(p.kappa1+p.kappa2*center)*(p.theta/center-1)+rate/center,
                                  p.kappa2*p.theta-p.kappa1-2*p.kappa2*center+rate/center,
                                  -p.kappa2*center])-z*p.beta*center*sigma2
                expected = polyadd(polymul(drift, polyder(a[q])), .5*p.vartheta2*polymul(
                    sigma2, polyadd(polyder(a[q], 2), polymul(polyder(a[q]), polyder(a[q])))))
                expected = polyadd(expected, .5*z*(z+1)*center**2*sigma2)[:degree+1]
                maximum_generator_error = max(maximum_generator_error, float(np.max(abs(expected-actual[q]))))
            new_center = center*1.15
            translated = experimental.translate_coefficients(a, center, new_center)
            points = np.array([.12, .26, .43])
            error = np.max(abs(polyval(points/center-1, a.T)-polyval(points/new_center-1, translated.T)))
            maximum_translation_error = max(maximum_translation_error, float(error))
            if degree == 4 and center == p.theta:
                for q, z in enumerate(phi):
                    physical = a[q]/(center**np.arange(degree+1))
                    m, ell, h = canonical.func_a_ode_quadratic_terms(
                        p.theta, p.kappa1, p.kappa2, p.beta, p.volvol, z, 0j,
                        True, canonical.ExpansionOrder.SECOND)
                    reference = (np.einsum('i,kij,j->k', physical, m, physical)+ell@physical+h)
                    reference *= center**np.arange(degree+1)
                    no_motion = experimental.polynomial_rhs(a[q].ravel(), phi[q:q+1], degree,
                        center, 0., p.theta, p.kappa1, p.kappa2, p.beta, p.volvol)
                    maximum_canonical_error = max(maximum_canonical_error, float(np.max(abs(reference-no_motion))))
    flat = LogSvParams(sigma0=.2, theta=.2, kappa1=3., kappa2=4., beta=0., volvol=0.)
    black, _ = experimental.solve_transform(flat, .7, phi, ExpansionConfig(rtol=1e-11, atol=1e-13))
    black_error = float(np.max(abs(black-np.exp(.5*.2**2*.7*phi*(phi+1)))))
    assert maximum_generator_error < 1e-10
    assert maximum_translation_error < 1e-11
    assert maximum_canonical_error < 1e-10
    assert black_error < 1e-10
    return dict(generator_max_error=maximum_generator_error,
                canonical_degree4_max_error=maximum_canonical_error,
                translation_max_error=maximum_translation_error, black_max_error=black_error)


def cases():
    """Return frozen parameters and actual maturity fractions, retaining chain IDs only as labels."""
    btc = LogSvParams(sigma0=.8626, theta=1.0418, kappa1=2.21, kappa2=2.18, beta=.1296, volvol=1.6286)
    chain = get_btc_test_chain_data()
    stress = LogSvParams(sigma0=.10690059522438819, theta=.1429987129151663,
                         kappa1=11.283811809963527, kappa2=.05000000557605693,
                         beta=-3.4324086406286938, volvol=3.)
    return [("btc_short_control", btc, float(chain.ttms[1]), str(chain.ids[1])),
            ("btc_longest_snapshot", btc, float(chain.ttms[-1]), str(chain.ids[-1])),
            ("btc_one_year", btc, 1., "synthetic 1y at same parameters"),
            ("difficult_spx_43d", stress, .11792237442922375, "43d")]


def configurations():
    """Prespecified changes, not a price-fitted search over centers or degrees."""
    return [
        ("degree4_theta", ExpansionConfig()),
        ("degree4_theta_loose", ExpansionConfig(rtol=1e-5, atol=1e-7)),
        ("degree4_theta_tight", ExpansionConfig(rtol=1e-11, atol=1e-13)),
        ("degree4_theta_16_segments", ExpansionConfig(segments=16)),
        ("degree2_theta", ExpansionConfig(degree=2)),
        ("degree6_theta", ExpansionConfig(degree=6)),
        ("degree8_theta", ExpansionConfig(degree=8)),
        ("degree4_sigma0", ExpansionConfig(center="sigma0")),
        ("degree4_characteristic", ExpansionConfig(center="characteristic")),
        ("degree4_translated4", ExpansionConfig(center="translated_characteristic", segments=4)),
        ("degree4_translated16", ExpansionConfig(center="translated_characteristic", segments=16)),
        ("degree6_characteristic", ExpansionConfig(degree=6, center="characteristic")),
        ("degree8_characteristic", ExpansionConfig(degree=8, center="characteristic")),
    ]


def benchmark(output):
    """Save each completed experiment, including rejected closures and integration budgets."""
    report = dict(algebra=algebra_checks(), price_units="undiscounted, forward-normalized calls",
                  interpretation="Local closure diagnostics, not a calibration recommendation",
                  source_sha256={Path(module.__file__).name: hashlib.sha256(
                      Path(module.__file__).read_bytes()).hexdigest()
                      for module in [experimental, canonical]}, cases=[])
    strikes = np.array([.95, 1., 1.05])
    for name, params, maturity, label in cases():
        reference_grid = PdeGrid(space_nodes=1601, time_steps=1024)
        reference = pde_calls(params, maturity, strikes, reference_grid)
        result = dict(name=name, maturity=maturity, days_365=maturity*365, chain_label=label,
                      parameters={key:float(getattr(params, key)) for key in
                                  ["sigma0", "theta", "kappa1", "kappa2", "beta", "volvol"]},
                      strikes_over_forward=strikes.tolist(), pde_prices=reference.tolist(),
                      pde_grid=asdict(reference_grid), variants=[])
        report["cases"].append(result)
        for variant, config in configurations():
            started = time.perf_counter()
            row = dict(variant=variant, config=asdict(config))
            try:
                prices, diagnostics = experimental.call_prices(params, maturity, strikes, config)
                error = (prices-reference)*1e4
                row.update(status="solved", prices=prices.tolist(), error_vs_pde_bp=error.tolist(),
                           max_absolute_error_bp=float(np.max(abs(error))), **diagnostics)
            except (FloatingPointError, ValueError, TimeoutError) as exc:
                row.update(status="rejected", reason=str(exc), seconds=time.perf_counter()-started)
            result["variants"].append(row)
            output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps(dict(case=name, maturity=maturity, **row)), flush=True)
    return report


class Locals(Enum):
    """Independent algebra checks and bounded numerical experiments."""

    ALGEBRA = "algebra"
    BENCHMARK = "benchmark"


def run_local(local: Locals, output=None):
    """Execute the selected experimental group."""
    if local is Locals.ALGEBRA:
        return algebra_checks()
    return benchmark(output)


def main():
    """Run from a checkout and write only the explicitly requested report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algebra-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.algebra_only:
        print(json.dumps(run_local(Locals.ALGEBRA)))
        return
    if args.output is None:
        parser.error("--output is required for the benchmark")
    target = args.output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    run_local(Locals.BENCHMARK, target)


if __name__ == "__main__":
    main()
