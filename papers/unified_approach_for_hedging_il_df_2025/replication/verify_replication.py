"""Verify theoretical IL replication against independent AMM inventories.

Run ``python verify_replication.py`` beside the paper replication modules.
No private SigmaStrats package, observed AMM data or volatility data are required.
The checks cover direct payoffs, expiry/boundaries, BSM quadrature and deltas,
static replication, nonzero carry via an explicit wrapper, and a constant-vol
limit of the LogSV Fourier pricer. Failures cause a nonzero exit status.
"""

import json

import numpy as np
from scipy.integrate import quad

import bsm_il
import il_payoffs
from run_logsv_for_il_payoff import logsv_il_pricer
from stochvolmodels import LogSvParams


def inventory_il(price, p0, pa, pb, borrowed):
    """Independent benchmark from AMM token balances, per unit initial notional."""
    p = np.asarray(price, dtype=float)
    c = np.clip(p, pa, pb)
    x, y = 1/np.sqrt(c)-1/np.sqrt(pb), np.sqrt(c)-np.sqrt(pa)
    x0, y0 = 1/np.sqrt(p0)-1/np.sqrt(pb), np.sqrt(p0)-np.sqrt(pa)
    v0 = p0*x0+y0
    return (p*x+y-(p*x0+y0 if borrowed else v0))/v0


def inventory_delta(price, p0, pa, pb, borrowed):
    """Derivative of the complete continuous AMM payoff, including boundaries."""
    x = 1/np.sqrt(np.clip(price, pa, pb))-1/np.sqrt(pb)
    x0, y0 = 1/np.sqrt(p0)-1/np.sqrt(pb), np.sqrt(p0)-np.sqrt(pa)
    return (x-x0 if borrowed else x)/(p0*x0+y0)


def quadrature(spot, p0, pa, pb, tau, sigma, borrowed, rate=0.0, yield_rate=0.0):
    """Discounted expectation of actual AMM inventory under a lognormal price."""
    forward = spot*np.exp((rate-yield_rate)*tau)
    width = sigma*np.sqrt(tau)
    if width == 0:
        return np.exp(-rate*tau)*inventory_il(forward, p0, pa, pb, borrowed)
    center = np.log(forward)-0.5*width**2
    cuts = sorted([-13.0, 13.0] + [z for k in (pa, pb)
                                 if -13 < (z := (np.log(k)-center)/width) < 13])

    def density_payoff(z):
        return (inventory_il(np.exp(center+width*z), p0, pa, pb, borrowed)
                * np.exp(-0.5*z*z)/np.sqrt(2*np.pi))

    return np.exp(-rate*tau)*sum(quad(density_payoff, a, b, epsabs=2e-11,
                                     epsrel=2e-11)[0]
                                for a, b in zip(cuts[:-1], cuts[1:]))


def run_checks():
    """Run independent deterministic checks and print a machine-readable summary."""
    results = []

    def check(name, errors, tolerance):
        error = float(np.max(np.abs(errors)))
        results.append(dict(check=name, comparisons=len(errors), max_error=error,
                            tolerance=tolerance, passed=error < tolerance))

    payoff, expiry, zero_vol, expiry_delta, value, delta, carry = [], [], [], [], [], [], []
    ranges = [(1.0, 0.9, 1.1), (2000.0, 1500.0, 2500.0),
              (60000.0, 30000.0, 90000.0), (2000.0, 1990.0, 2020.0)]
    for p0, pa, pb in ranges:
        grid = np.unique(np.r_[np.geomspace(0.25*pa, 3*pb, 101), pa, p0, pb])
        for borrowed in (False, True):
            reference = inventory_il(grid, p0, pa, pb, borrowed)
            payoff.extend(il_payoffs.il_payoff(grid, p0, pa, pb, borrowed=borrowed)-reference)
            payoff.extend(il_payoffs.il_decomposition(grid, p0, pa, pb,
                                                      borrowed=borrowed)-reference)
            for spot in (0.8*pa, pa, p0, pb, 1.2*pb):
                reference = inventory_il(spot, p0, pa, pb, borrowed)
                expiry.append(bsm_il.compute_bsm_il_value(p0, spot, pa, pb, 0.0, 0.6,
                                                          1.0, borrowed)-reference)
                zero_vol.append(bsm_il.compute_bsm_il_value(p0, spot, pa, pb, 0.7, 0.0,
                                                            1.0, borrowed)-reference)
                expiry_delta.append(p0*(bsm_il.compute_bsm_il_delta(
                    p0, spot, pa, pb, 0.0, 0.6, 1.0, borrowed)
                    - inventory_delta(spot, p0, pa, pb, borrowed)))
                for tau, sigma in ((1/365, 0.2), (14/365, 0.6), (0.5, 1.2)):
                    actual = bsm_il.compute_bsm_il_value(p0, spot, pa, pb, tau, sigma,
                                                        1.0, borrowed)
                    value.append(actual-quadrature(spot, p0, pa, pb, tau, sigma, borrowed))
                    h = 1e-5*spot
                    fd = (bsm_il.compute_bsm_il_value(p0, spot+h, pa, pb, tau, sigma,
                                                     1.0, borrowed)
                          - bsm_il.compute_bsm_il_value(p0, spot-h, pa, pb, tau, sigma,
                                                       1.0, borrowed))/(2*h)
                    delta.append(p0*(bsm_il.compute_bsm_il_delta(
                        p0, spot, pa, pb, tau, sigma, 1.0, borrowed)-fd))
                    for rate, yield_rate in ((0.07, 0.02), (-0.01, 0.03)):
                        forward = spot*np.exp((rate-yield_rate)*tau)
                        wrapped = np.exp(-rate*tau)*bsm_il.compute_bsm_il_value(
                            p0, forward, pa, pb, tau, sigma, 1.0, borrowed)
                        carry.append(wrapped-quadrature(spot, p0, pa, pb, tau, sigma,
                                                        borrowed, rate, yield_rate))
    check("V3 direct and decomposed payoffs vs inventory", payoff, 2e-10)
    check("exact expiry and range boundaries", expiry, 2e-10)
    check("zero volatility and range boundaries", zero_vol, 2e-10)
    check("expiry delta (p0 scaled)", expiry_delta, 1e-10)
    check("BSM vs inventory quadrature", value, 2e-9)
    check("BSM delta vs finite differences (p0 scaled)", delta, 3e-7)
    check("nonzero-carry forward/discount wrapper", carry, 2e-9)

    v2 = []
    for p0 in (1.0, 2000.0, 60000.0):
        p = p0*np.geomspace(0.1, 10.0, 101)
        for borrowed in (False, True):
            reference = np.sqrt(p/p0)-(0.5*(p/p0+1) if borrowed else 1)
            v2.extend(il_payoffs.il_payoff(p, p0, borrowed=borrowed)-reference)
    check("V2 payoffs", v2, 1e-12)

    p0, pa, pb = 2000.0, 1500.0, 2500.0
    puts, calls = np.arange(pa, p0+1, 50), np.arange(p0, pb+1, 50)
    wp, wc = il_payoffs.static_hedge(p0, pa, pb, puts, calls)
    knots = np.unique(np.r_[puts, calls])
    strip = il_payoffs.hedge_payoff(knots, puts, calls, wp, wc)[2]
    check("static hedge exact at every strike", strip+inventory_il(knots, p0, pa, pb, True), 1e-12)
    fine = np.linspace(pa, pb, 10001)
    strip = il_payoffs.hedge_payoff(fine, puts, calls, wp, wc)[2]
    residual_bp = float(np.max(np.abs(strip+inventory_il(fine, p0, pa, pb, True)))*1e4)
    check("published 50-dollar-grid residual below 2.5 bp", [residual_bp], 2.5)

    limit = []
    for forward, tau, sigma in ((1400.0, 0.1, 0.4), (2000.0, 14/365, 0.6),
                                (2700.0, 0.25, 0.8)):
        params = LogSvParams(sigma0=sigma, theta=sigma, kappa1=2.0, kappa2=2.0,
                             beta=0.0, volvol=1e-5)
        actual = logsv_il_pricer(params, tau, forward, p0, pa, pb, notional=1.0)
        limit.append(actual-quadrature(forward, p0, pa, pb, tau, sigma, True))
    check("constant-vol LogSV vs direct inventory quadrature", limit, 1e-7)

    # Deliberately reintroduce the missing factor 2 at entry: this must fail.
    _, _, liquidity = il_payoffs.initial_position(p0, pa=pa, pb=pb)
    mutated_missing_factor = -liquidity*np.sqrt(p0)
    mutation_detected = bool(abs(mutated_missing_factor-inventory_il(p0, p0, pa, pb, True)) > 1)
    old_printed_value = -4*np.sqrt(p0)
    output = dict(checks=results, all_passed=all(row["passed"] for row in results),
                  static_replication_error_bp=residual_bp,
                  mutation_check=dict(missing_factor_detected=mutation_detected,
                                      original_printed_expiry_value=float(old_printed_value),
                                      correct_expiry_value=0.0),
                  conventions="P&L sign; unit initial quote notional; forward-valued BSM; external discounting")
    print(json.dumps(output, indent=2))
    assert output["all_passed"] and mutation_detected, "IL replication verification failed"
    return output


if __name__ == "__main__":
    run_checks()
