"""Inspect retained and discarded generator terms without altering production pricing.

The conditional Lewis transform obeys |u|<=1 under the MMA martingale condition.
A finite price and that bound at sigma0 do not control a polynomial approximant
at volatility states subsequently visited by the diffusion. Residuals here are
diagnostics only: no weighted global error bound is asserted.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.polynomial.polynomial import polyadd, polyder, polymul, polyval
from scipy.special import comb

from . import expansion_diagnostics as experimental
from .expansion_diagnostics import ExpansionConfig
from .expansion_diagnostics_run import algebra_checks, canonical, cases
from .logvol_pde import PdeGrid, logvol_transform


def validation_checks():
    """Prove detection of the historical coefficient defect and check the moving-coordinate sign."""
    passed = algebra_checks()
    original = canonical.func_a_ode_quadratic_terms

    def historical_diffusion_entry(*args, **kwargs):
        m, ell, h = original(*args, **kwargs)
        ell[4, 4] -= 4*(args[3]**2+args[4]**2)
        return m, ell, h

    with patch.object(canonical, "func_a_ode_quadratic_terms", historical_diffusion_entry):
        try:
            algebra_checks()
        except AssertionError:
            defect_detected = True
        else:
            defect_detected = False
    assert defect_detected

    # A time-independent physical polynomial G(sigma) acquires coordinate motion
    # A(tau,x)=G(c(tau)*(1+x)). Its independent finite difference must agree with
    # +(c'/c)*(1+x)*A_x when the physical generator is zero (phi=0,k1=k2=v2=0).
    physical = np.array([.3, -.7, .9, -.2, .4], complex)
    center, rate, delta = .37, -.08, 1e-6

    def coefficients(c):
        return np.array([sum(physical[j]*c**j*comb(j, k, exact=True)
                             for j in range(k, len(physical))) for k in range(len(physical))])

    expected = (coefficients(center+rate*delta)-coefficients(center-rate*delta))/(2*delta)
    got = experimental.polynomial_rhs(coefficients(center), np.array([0j]), 4, center,
                                       rate, 1., 0., 0., 0., 0.)
    coordinate_error = float(np.max(abs(got-expected)))
    assert coordinate_error < 1e-9
    return dict(**passed, historical_diffusion_defect_detected=defect_detected,
                moving_coordinate_chain_rule_error=coordinate_error)


def inspect_case(name, params, maturity, label):
    """Capture the genuine integrated state, then evaluate its unprojected PDE residual."""
    phi = -.5+1j*np.array([0., 1., 5., 20.])
    captured = {}
    original = experimental.solve_ivp

    def capture(*args, **kwargs):
        solution = original(*args, **kwargs)
        captured["solution"] = solution
        return solution

    config = ExpansionConfig(rtol=1e-11, atol=1e-13)
    with patch.object(experimental, "solve_ivp", capture):
        transform, diagnostics = experimental.solve_transform(params, maturity, phi, config)
    a = captured["solution"].y[:, -1].reshape((len(phi), 5))
    c = params.theta
    # A prespecified state interval, not tuned against the reference transform.
    sigma = params.sigma0*np.geomspace(.25, 4., 41)
    x = sigma/c-1
    state0 = params.sigma0/c-1
    reference = logvol_transform(params, maturity, phi, PdeGrid(space_nodes=1601, time_steps=1024))
    entries = []
    for index, z in enumerate(phi):
        s2 = np.array([1., 2., 1.])
        drift = np.array([(params.kappa1+params.kappa2*c)*(params.theta/c-1),
                          params.kappa2*params.theta-params.kappa1-2*params.kappa2*c,
                          -params.kappa2*c])-z*params.beta*c*s2
        derivative = polyder(a[index])
        generator = polyadd(polymul(drift, derivative), .5*params.vartheta2*polymul(
            s2, polyadd(polyder(a[index], 2), polymul(derivative, derivative))))
        generator = polyadd(generator, .5*z*(z+1)*c*c*s2)
        omitted = generator.copy()
        omitted[:5] = 0
        log_u = polyval(x, a[index])
        omitted_at_state = polyval(x, omitted)
        entries.append(dict(
            frequency=float(z.imag), transform_error_at_sigma0=float(abs(transform[index]-reference[index])),
            leading_coefficient_real=float(a[index, -1].real),
            max_log_modulus_on_state_grid=float(np.max(log_u.real)),
            count_state_modulus_violations=int(np.count_nonzero(log_u.real > 1e-8)),
            omitted_generator_at_sigma0_abs=float(abs(polyval(state0, omitted))),
            max_omitted_generator_on_state_grid_abs=float(np.max(abs(omitted_at_state))),
            log_modulus_on_state_grid=log_u.real.tolist()))
    growth = {str(n): float(.5*n*(n-1)*params.vartheta2
                           -n*(params.kappa1+params.kappa2*params.theta)+n*params.beta*params.theta)
              for n in range(2, 9)}
    return dict(name=name, maturity=maturity, days_365=maturity*365, chain_label=label,
                linear_top_degree_coefficient_real=growth, sigma_state_grid=sigma.tolist(),
                solved_diagnostics=diagnostics, frequencies=entries)


def main():
    """Save the explicitly requested mathematical diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    target = args.output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    report = dict(validation=validation_checks(),
                  source_sha256=hashlib.sha256(Path(experimental.__file__).read_bytes()).hexdigest(),
                  interpretation="The retained highest-degree coefficient is not a full Jacobian eigenvalue; omitted-term magnitudes are not price-error bounds",
                  cases=[])
    for name, params, maturity, label in cases():
        report["cases"].append(inspect_case(name, params, maturity, label))
        target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(name, "finished", flush=True)


if __name__ == "__main__":
    main()
