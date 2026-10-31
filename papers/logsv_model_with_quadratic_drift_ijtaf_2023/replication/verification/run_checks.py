"""Offline validation of corrected LogSV equations and their approximation limits.

Run from the repository root with ``python -m
papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.run_checks``.
The full run compares the independent state-grid PDE with canonical Monte Carlo
and reports finite-expansion error. A large expansion error at the deliberately
difficult SPX case is an observed limitation, not a failed PDE verification.
No market-data downloads or private datasets are used.
"""
import argparse
from dataclasses import asdict
from enum import Enum
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numba import njit
from numpy.polynomial.polynomial import polyadd, polyder, polymul
from scipy.integrate import quad, solve_ivp

from stochvolmodels import LogSvParams, local_path
from stochvolmodels.pricers.logsv import affine_expansion as afe
from .fourier import calls_from_transform, gaussian_grid
from .logvol_pde import PdeGrid, call_prices as pde_calls, logvol_transform
from .monte_carlo import bridge_normals, call_prices as mc_calls


def coefficient_checks():
    """Expand the generator independently of the implementation's matrix tables."""
    rng = np.random.default_rng(20261029)
    errors = []
    for degree, order in [(2, afe.ExpansionOrder.FIRST), (4, afe.ExpansionOrder.SECOND)]:
        for spot in [True, False]:
            for eta in [.8, 1., 1.2]:
                for _ in range(8):
                    theta, k1, k2 = rng.uniform(.1, .8), rng.uniform(.5, 8), rng.uniform(.5, 8)
                    beta, eps = rng.uniform(-3, 3), rng.uniform(.3, 3)
                    phi = (-.5 if spot else .5)+1j*rng.uniform(0, 8)
                    psi = .03-.02j
                    v2 = beta*beta+eps*eps
                    lam = 0. if spot else beta*theta**2*eta
                    kp = k1+k2*theta-(0. if spot else 2*beta*theta*eta)
                    k2p = k2-(0. if spot else beta*eta)
                    s2 = np.array([theta**2, 2*theta, 1.])
                    drift = np.array([lam, -kp, -k2p])-beta*eta*phi*s2
                    a = rng.normal(size=degree+1)+1j*rng.normal(size=degree+1)
                    full = polyadd(polymul(drift, polyder(a)), .5*v2*polymul(
                        s2, polyadd(polyder(a, 2), polymul(polyder(a), polyder(a)))))
                    full = polyadd(full, .5*eta**2*(phi*(phi+(1 if spot else -1))-2*psi)*s2)
                    m, ell, h = afe.func_a_ode_quadratic_terms(
                        theta, k1, k2, beta, eps, phi, psi, spot, order, eta)
                    actual = np.einsum('i,kij,j->k', a, m, a)+ell@a+h
                    errors.append(float(np.max(abs(actual-full[:degree+1]))))
    maximum = max(errors)
    if maximum > 1e-10:
        raise AssertionError(f"Canonical coefficient tensors disagree with generator: {maximum}")
    return dict(cases=len(errors), max_error=maximum, tolerance=1e-10)


def stationary_moment_checks():
    """Integrate the stationary density directly, without its Bessel moment formula."""
    from ..theory.steady_state_pdf import vol_moment

    cases = [(0.37, 3.0, 5.0), (0.8, 1.7, 0.6), (0.37, 3.0, 0.0)]
    rows = []
    for theta, kappa1, kappa2 in cases:
        params = LogSvParams(theta=theta, kappa1=kappa1, kappa2=kappa2,
                            beta=-0.25, volvol=0.8)
        variance = params.beta**2 + params.volvol**2
        nu = 2*(kappa2*theta-kappa1)/variance - 1
        q, b = 2*kappa1*theta/variance, 2*kappa2/variance

        # sigma=exp(y) includes the Jacobian. These finite log-volatility bounds
        # are far beyond the density tails in the specified cases; the inverse-
        # gamma case has alpha > 9 and the highest tested moment is four.
        def integral(r):
            return quad(lambda y: np.exp((nu+r)*y-q*np.exp(-y)-b*np.exp(y)),
                        -40., 40., epsabs=1e-13, epsrel=1e-11, limit=200)[0]

        normalizer = integral(0)
        for r in [0, 1, 2, 3, 4]:
            expected = integral(r)/normalizer
            actual = float(vol_moment(params, r))
            error = abs(actual-expected)/expected
            assert np.isfinite(actual) and error < 1e-9, (
                theta, kappa1, kappa2, r, actual, expected, error)
            rows.append(dict(theta=theta, kappa1=kappa1, kappa2=kappa2, order=r,
                             formula=actual, quadrature=expected, relative_error=error))
    # The inverse-gamma tail admits order r iff r < alpha = 1+2*kappa1/vartheta^2.
    heavy_tail = LogSvParams(theta=.37, kappa1=.5, kappa2=0., beta=0., volvol=1.)
    assert np.isclose(vol_moment(heavy_tail, 1), .37, rtol=1e-14)
    assert np.isposinf(vol_moment(heavy_tail, 2))
    assert np.isposinf(vol_moment(heavy_tail, 3))
    return dict(cases=len(rows), max_relative_error=max(row['relative_error'] for row in rows),
                tolerance=1e-9, divergent_inverse_gamma_moments='positive infinity', rows=rows)


def black_and_endpoint_checks():
    """Check time convergence against the analytic constant-volatility transform."""
    p = LogSvParams(sigma0=.2, theta=.2, kappa1=2., kappa2=4., beta=0., volvol=0.)
    phi = -.5+1j*np.array([0., 1., 5., 20., 40.])
    expected = np.exp(.5*.2**2*.25*phi*(phi+1))
    errors = []
    for steps in [256, 1024]:
        got = logvol_transform(p, .25, phi, PdeGrid(time_steps=steps))
        errors.append(float(np.max(abs(got-expected))))
    assert errors[1] < errors[0]/8 and errors[1] < 1e-7
    endpoints = logvol_transform(p, .25, np.array([0j, -1+0j]))
    assert np.array_equal(endpoints, np.ones(2))
    assert np.array_equal(logvol_transform(p, 0., phi), np.ones(len(phi)))
    return dict(time_steps=[256, 1024], transform_max_errors=errors,
                endpoints='Exact identities handled separately; not a PDE boundary test')


def linearized_ode_check():
    """Check the paper diagnostic's singular linear system against direct integration."""
    from ..theory.ode_sol_in_time import solve_approximate_solutions
    p = LogSvParams(sigma0=.227, theta=.2616, kappa1=4.9325, kappa2=18.855,
                    beta=-1.8123, volvol=.9832)
    args = dict(theta=p.theta, kappa1=p.kappa1, kappa2=p.kappa2, beta=p.beta,
                volvol=p.volvol, phi=-.5+2j, psi=0j, expansion_order=afe.ExpansionOrder.SECOND)
    _, ell, h = afe.func_a_ode_quadratic_terms(**args)
    assert np.max(abs(solve_approximate_solutions(ttm=0., **args))) == 0
    direct = solve_ivp(lambda t, a: ell@a+h, (0., .2), np.zeros(5, complex),
                       method='DOP853', rtol=1e-11, atol=1e-13)
    error = float(np.max(abs(solve_approximate_solutions(ttm=.2, **args)-direct.y[:, -1])))
    assert error < 1e-9
    return dict(max_error=error, tolerance=1e-9)


@njit(cache=False)
def _rhs(flat, m, ell, h):
    a = flat.reshape((-1, 5))
    out = np.empty_like(a)
    for q in range(len(a)):
        for k in range(5):
            out[q, k] = h[q, k]
            for i in range(5):
                out[q, k] += ell[q, k, i]*a[q, i]
                for j in range(5):
                    out[q, k] += m[q, k, i, j]*a[q, i]*a[q, j]
    return out.ravel()


def expansion_calls(params, maturity, strikes):
    """Tightly integrate the canonical corrected tensors for diagnostic comparison."""
    phi, weights = gaussian_grid()
    tensors = [afe.func_a_ode_quadratic_terms(
        params.theta, params.kappa1, params.kappa2, params.beta, params.volvol, z, 0j,
        True, afe.ExpansionOrder.SECOND) for z in phi]
    m, ell, h = (np.array([row[i] for row in tensors]) for i in range(3))
    initial = np.zeros(5*len(phi), complex)
    _rhs(initial, m, ell, h)  # exclude JIT startup from the solve budget
    started = time.perf_counter()

    def rhs(t, a):
        if time.perf_counter()-started > 30:
            raise TimeoutError("Coefficient solve exceeded 30 seconds.")
        return _rhs(a, m, ell, h)

    def growth(t, a):
        return 1e9-np.max(abs(a))

    growth.terminal = True
    sol = solve_ivp(rhs, (0, maturity), initial, method='DOP853', rtol=1e-9,
                    atol=1e-11, events=growth)
    if not sol.success or sol.t[-1] < maturity-1e-12:
        raise FloatingPointError(f"Coefficient solve stopped at {sol.t[-1]} before {maturity}.")
    lm = sol.y[:, -1].reshape((-1, 5))@((params.sigma0-params.theta)**np.arange(5))
    transform = np.exp(lm)
    return calls_from_transform(transform, phi, weights, maturity, params.sigma0, strikes)


def pricing_checks():
    """Compare PDE grids and independent simulation; expose expansion errors separately."""
    spy = LogSvParams(sigma0=.2270, theta=.2616, kappa1=4.9325, kappa2=18.8550,
                      beta=-1.8123, volvol=.9832)
    btc = LogSvParams(sigma0=.8327, theta=1.0139, kappa1=4.8609, kappa2=4.7940,
                      beta=.1988, volvol=2.3694)
    old_spx = LogSvParams(sigma0=.10690059522438819, theta=.1429987129151663,
                          kappa1=11.283811809963527, kappa2=.05000000557605693,
                          beta=-3.4324086406286938, volvol=3.)
    cases = [('published_spy_12d', spy, 12/365), ('published_spy_93d', spy, 93/365),
             ('published_btc_1m', btc, 1/12), ('difficult_spx_43d', old_spx, .11792237442922375)]
    strikes = np.array([.95, 1., 1.05])
    rows = []
    for name, params, maturity in cases:
        coarse_grid = PdeGrid(space_nodes=801, time_steps=512)
        fine_grid = PdeGrid(space_nodes=1601, time_steps=1024)
        coarse = pde_calls(params, maturity, strikes, coarse_grid)
        fine = pde_calls(params, maturity, strikes, fine_grid)
        refinement = abs(fine-coarse)
        assert np.max(refinement) < 1e-5, (name, refinement)
        fine_mc, coarse_mc = [], []
        for seed in [291026, 291027, 291028, 291029]:
            z = bridge_normals(power=15, steps=512, seed=seed)
            fine_mc.append(mc_calls(params, maturity, strikes, z))
            coarse_z = np.ascontiguousarray(z.reshape((256, 2, -1)).sum(axis=1)/np.sqrt(2))
            coarse_mc.append(mc_calls(params, maturity, strikes, coarse_z))
        mean = np.mean(fine_mc, axis=0)
        se = np.std(fine_mc, axis=0, ddof=1)/2
        mesh_change = abs(mean-np.mean(coarse_mc, axis=0))
        allowance = 4*se+2*mesh_change+refinement+5e-7
        assert np.all(abs(fine-mean) < allowance), (name, fine-mean, allowance)
        row = dict(case=name, parameters={k:float(getattr(params, k)) for k in
                   ['sigma0','theta','kappa1','kappa2','beta','volvol']}, maturity=maturity,
                   strikes_over_forward=strikes.tolist(), pde_grid=asdict(fine_grid),
                   pde_prices=fine.tolist(), pde_refinement_bp=(refinement*1e4).tolist(),
                   mc_prices=mean.tolist(), mc_scramble_se_bp=(se*1e4).tolist(),
                   mc_time_refinement_bp=(mesh_change*1e4).tolist(),
                   pde_minus_mc_bp=((fine-mean)*1e4).tolist(),
                   numerical_comparison_allowance_bp=(allowance*1e4).tolist(), reference_passed=True)
        try:
            expanded = expansion_calls(params, maturity, strikes)
            row['expansion'] = dict(status='solved', prices=expanded.tolist(),
                                     error_vs_pde_bp=((expanded-fine)*1e4).tolist())
        except (FloatingPointError, TimeoutError, ValueError) as exc:
            row['expansion'] = dict(status='rejected', reason=str(exc))
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


class Locals(Enum):
    """Offline verification groups."""

    COEFFICIENTS = 'coefficients'
    BLACK = 'black'
    LINEARIZED = 'linearized'
    STATIONARY = 'stationary'
    PRICING = 'pricing'


def run_local(local: Locals):
    """Execute one group; numerical-reference failures raise an exception."""
    return {Locals.COEFFICIENTS: coefficient_checks, Locals.BLACK: black_and_endpoint_checks,
            Locals.LINEARIZED: linearized_ode_check, Locals.STATIONARY: stationary_moment_checks,
            Locals.PRICING: pricing_checks}[local]()


def main():
    """Write reproducible numerical evidence to a requested output file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='Skip the slower PDE/MC price checks.')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    target = args.output or Path(local_path.get_output_path())/'logsv_paper_verification.json'
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    result = dict(scope='MMA flat-backbone reference pricing; both-measure coefficient algebra',
                  source_sha256=hashlib.sha256(Path(afe.__file__).read_bytes()).hexdigest(),
                  simulation=dict(scrambles=4, power=15, fine_steps=512, coarse_steps=256,
                                  seeds=[291026,291027,291028,291029]), checks={})
    for local in Locals:
        if args.quick and local is Locals.PRICING:
            continue
        result['checks'][local.value] = run_local(local)
        target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        print(f'{local.value}: passed', flush=True)
    print(f'Verification saved: {target}', flush=True)


if __name__ == '__main__':
    main()
