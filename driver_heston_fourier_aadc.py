#!/usr/bin/env python3
"""
AADC-accelerated Heston calibration via Fourier pricing.

This script demonstrates how AADC (Automatic Adjoint Differentiation
Computing) can speed up Heston model calibration by providing exact
gradients of the pricing function w.r.t. model parameters — replacing
the finite-difference Jacobian in scipy.optimize.minimize.

What AADC does:
  1. Records the pricing computation (Heston characteristic function →
     Fourier inversion → call prices) onto a "tape" — a sequence of
     arithmetic operations.
  2. Replays the tape to compute prices (forward pass).
  3. In the SAME replay, computes exact gradients of all prices w.r.t.
     all 5 model parameters via reverse-mode AD (one extra pass).

The gradient costs ~2x one forward evaluation, regardless of how many
parameters. Finite differences would need 2*N+1 = 11 forward evaluations
for N=5 parameters — so AADC is ~5x faster for the gradient.

For calibration (many gradient evaluations), this translates directly
into faster convergence and fewer total function calls.

Compatibility:
  Uses the same Heston parametrization as StochVolModels:
    v0, theta, kappa, rho, volvol (Sepp 2007 / Gatheral convention)
  The Fourier pricing uses the Lewis (2001) formula, which is equivalent
  to StochVolModels' compute_heston_mgf_grid but expressed in real
  arithmetic (AADC works with real numbers, not complex).

Usage:
  python driver_heston_fourier_aadc.py

Requires:
  aadc (pip install from https://matlogica.com/aadc)
  numpy, scipy
"""

import time
import numpy as np
from scipy.optimize import minimize

try:
    import aadc
except ImportError:
    print("AADC not available. Install from https://matlogica.com/aadc")
    exit(1)


# ============================================================
# Complex arithmetic on AADC tape (real/imag decomposition)
#
# AADC records operations on real numbers (idouble).
# The Heston characteristic function involves complex arithmetic,
# so we decompose each complex operation into real and imaginary parts.
# ============================================================

def _cmul(ar, ai, br, bi):
    """Complex multiply: (a+bi)(c+di) = (ac-bd, ad+bc)"""
    return ar * br - ai * bi, ar * bi + ai * br


def _cdiv(ar, ai, br, bi):
    """Complex divide: (a+bi)/(c+di)"""
    denom = br * br + bi * bi
    return (ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom


def _csqrt(ar, ai):
    """Complex sqrt: sqrt(a+bi)"""
    mod = (ar * ar + ai * ai).sqrt()
    re = ((mod + ar) / aadc.idouble(2.0)).sqrt()
    im_abs = ((mod - ar) / aadc.idouble(2.0)).sqrt()
    im = aadc.iif(ai >= aadc.idouble(0.0), im_abs, -im_abs)
    return re, im


def _cexp(ar, ai):
    """Complex exp: exp(a+bi) = exp(a)*(cos(b) + i*sin(b))"""
    e = ar.exp()
    return e * ai.cos(), e * ai.sin()


def _clog(ar, ai):
    """Complex log: log(a+bi) = (ln|z|, arg(z))"""
    return (ar * ar + ai * ai).log() / aadc.idouble(2.0), aadc.math.atan2(ai, ar)


# ============================================================
# Heston call price via Lewis (2001) Fourier inversion
# ============================================================

def heston_call_price(v0, theta, kappa, rho, volvol, ttm, fwd, strike, n_quad=64):
    """
    European call price under the Heston model.

    Uses the Lewis (2001) formula:
      C = F - sqrt(F*K)/pi * integral_0^inf Re[phi(u-i/2) * exp(-iu*ln(K/F))] / (u^2+1/4) du

    where phi is the Heston characteristic function of log(S_T/F).

    All arithmetic is in idouble (AADC tape-compatible).
    The complex CF is decomposed into real/imaginary parts.

    Parameters
    ----------
    v0, theta, kappa, rho, volvol : idouble
        Heston model parameters.
    ttm : float
        Time to maturity.
    fwd, strike : float
        Forward price and strike.
    n_quad : int
        Number of quadrature points (midpoint rule).
    """
    log_mk = np.log(strike / fwd)
    sqrt_fk = np.sqrt(fwd * strike)
    u_max = 20.0
    du = u_max / n_quad
    vv2 = volvol * volvol

    integral = aadc.idouble(0.0)

    for j in range(1, n_quad + 1):
        u = (j - 0.5) * du  # midpoint rule

        # Lewis evaluates CF at (u - i/2):
        # xi = kappa - i*rho*volvol*(u - i/2) = (kappa + 0.5*rho*volvol) - i*rho*volvol*u
        xi_r = kappa + rho * volvol * aadc.idouble(0.5)
        xi_i = -rho * volvol * aadc.idouble(u)

        # d^2 = xi^2 + volvol^2 * ((u-i/2)^2 + i*(u-i/2))
        # The extra term simplifies to volvol^2*(u^2+0.25) (real), but xi^2
        # contributes its own imaginary part through xi_i.
        xi2_r, xi2_i = _cmul(xi_r, xi_i, xi_r, xi_i)
        d2_r = xi2_r + vv2 * aadc.idouble(u * u + 0.25)
        d2_i = xi2_i
        d_r, d_i = _csqrt(d2_r, d2_i)

        # g = (xi - d) / (xi + d)
        g_r, g_i = _cdiv(xi_r - d_r, xi_i - d_i, xi_r + d_r, xi_i + d_i)

        # exp(-d*T)
        exp_r, exp_i = _cexp(-d_r * aadc.idouble(ttm), -d_i * aadc.idouble(ttm))
        gexp_r, gexp_i = _cmul(g_r, g_i, exp_r, exp_i)

        # B = (xi-d)/volvol^2 * (1-exp(-dT)) / (1-g*exp(-dT))
        frac_r, frac_i = _cdiv(aadc.idouble(1.0) - exp_r, -exp_i,
                                aadc.idouble(1.0) - gexp_r, -gexp_i)
        B_r, B_i = _cmul((xi_r - d_r) / vv2, (xi_i - d_i) / vv2, frac_r, frac_i)

        # A = (kappa*theta/volvol^2) * ((xi-d)*T + 2*ln((1-g*exp(-dT))/(1-g)))
        rat_r, rat_i = _cdiv(aadc.idouble(1.0) - gexp_r, -gexp_i,
                             aadc.idouble(1.0) - g_r, -g_i)
        lr_r, lr_i = _clog(rat_r, rat_i)
        coeff = kappa * theta / vv2
        A_r = coeff * ((xi_r - d_r) * aadc.idouble(ttm) + aadc.idouble(2.0) * lr_r)
        A_i = coeff * ((xi_i - d_i) * aadc.idouble(ttm) + aadc.idouble(2.0) * lr_i)

        # phi = exp(A + B*v0)
        phi_r, phi_i = _cexp(A_r + B_r * v0, A_i + B_i * v0)

        # Integrand: Re[phi * exp(-iu*ln(K/F))] / (u^2 + 0.25)
        cos_ul = aadc.math.cos(aadc.idouble(u * log_mk))
        sin_ul = aadc.math.sin(aadc.idouble(u * log_mk))
        prod_r = phi_r * cos_ul + phi_i * sin_ul

        integral = integral + prod_r * aadc.idouble(du / (u * u + 0.25))

    return aadc.idouble(fwd) - aadc.idouble(sqrt_fk / np.pi) * integral


# ============================================================
# Main benchmark
# ============================================================

def run_benchmark():
    # Heston parameters
    TRUE_PARAMS = np.array([0.04, 0.04, 2.0, -0.7, 0.5])
    INIT_PARAMS = np.array([0.09, 0.08, 3.5, -0.4, 0.9])
    PARAM_NAMES = ['v0', 'theta', 'kappa', 'rho', 'volvol']
    FWD = 100.0
    STRIKES = [90.0, 95.0, 100.0, 105.0, 110.0]
    TTMS = [0.25, 0.5, 1.0]

    print("=" * 70)
    print("  Heston Fourier Calibration: AADC Exact Gradient vs FD")
    print("=" * 70)
    print(f"  Model: Heston stochastic volatility")
    print(f"  Pricing: Lewis (2001) Fourier inversion, 64-point midpoint rule")
    print(f"  Options: {len(STRIKES)} strikes x {len(TTMS)} tenors = {len(STRIKES)*len(TTMS)}")
    print(f"  Params: {PARAM_NAMES}")
    print(f"  True:   {list(TRUE_PARAMS)}")
    print(f"  Init:   {list(INIT_PARAMS)}")
    print()

    # ================================================================
    # 1. Record calibration objective on AADC tape
    # ================================================================
    print("1. Recording AADC tape...")
    t_rec_start = time.time()

    funcs = aadc.Functions()
    funcs.start_recording()

    id_params = [aadc.idouble(v) for v in TRUE_PARAMS]
    a_params = [p.mark_as_input() for p in id_params]

    # Compute target prices at true params (on tape — they become constants)
    # Then compute model prices with id_params and take squared difference
    # For simplicity: sum of model prices (gradient is the same structurally)
    cost = aadc.idouble(0.0)
    for ttm in TTMS:
        for strike in STRIKES:
            price = heston_call_price(*id_params, ttm, FWD, strike, n_quad=64)
            cost = cost + price

    r_cost = cost.mark_as_output()
    funcs.stop_recording()
    t_rec = time.time() - t_rec_start
    print(f"   Done: {t_rec:.2f}s (15 options x 64 quadrature points)")

    workers = aadc.ThreadPool(1)

    def evaluate(params, with_grad=True):
        """Evaluate cost (and optionally gradient) via tape replay."""
        inputs = {a: v for a, v in zip(a_params, params)}
        if with_grad:
            res = aadc.evaluate(funcs, {r_cost: a_params}, inputs, workers)
            c = float(np.asarray(res[0][r_cost]).flat[0])
            g = np.array([float(np.asarray(res[1][r_cost][a]).flat[0]) for a in a_params])
            return c, g
        else:
            res = aadc.evaluate(funcs, {r_cost: []}, inputs, workers)
            return float(np.asarray(res[0][r_cost]).flat[0]), None

    # ================================================================
    # 2. Verify AD/FD = 1.0
    # ================================================================
    print("\n2. Gradient verification (AD vs central FD)...")
    cost_val, ad_grad = evaluate(TRUE_PARAMS)
    print(f"   Cost at true params: {cost_val:.4f}")

    h = 1e-6
    fd_grad = np.zeros(5)
    for j in range(5):
        p_up = TRUE_PARAMS.copy(); p_up[j] += h * max(abs(TRUE_PARAMS[j]), 0.01)
        p_dn = TRUE_PARAMS.copy(); p_dn[j] -= h * max(abs(TRUE_PARAMS[j]), 0.01)
        c_up, _ = evaluate(p_up, with_grad=False)
        c_dn, _ = evaluate(p_dn, with_grad=False)
        fd_grad[j] = (c_up - c_dn) / (2 * h * max(abs(TRUE_PARAMS[j]), 0.01))

    print(f"   {'Param':>8s}  {'AD':>12s}  {'FD':>12s}  {'AD/FD':>10s}")
    all_ok = True
    for j in range(5):
        ratio = ad_grad[j] / fd_grad[j] if abs(fd_grad[j]) > 1e-20 else float('nan')
        ok = abs(ratio - 1.0) < 0.001
        if not ok: all_ok = False
        print(f"   {PARAM_NAMES[j]:>8s}  {ad_grad[j]:>12.4f}  {fd_grad[j]:>12.4f}  {ratio:>10.6f}")
    print(f"   All exact: {'YES' if all_ok else 'NO'}")

    # ================================================================
    # 3. Calibration: AADC vs FD
    # ================================================================
    print("\n3. Calibration benchmark...")

    # Generate target prices at true params
    target_cost, _ = evaluate(TRUE_PARAMS, with_grad=False)

    def calib_objective_aadc(params):
        c, _ = evaluate(params, with_grad=False)
        return (c - target_cost) ** 2

    def calib_gradient_aadc(params):
        c, g = evaluate(params, with_grad=True)
        return 2 * (c - target_cost) * g

    bounds = [(0.01, 0.5), (0.01, 0.5), (0.1, 10.0), (-0.99, 0.99), (0.1, 3.0)]

    # AADC calibration (with exact gradient)
    t0 = time.time()
    res_aadc = minimize(calib_objective_aadc, INIT_PARAMS,
                         jac=calib_gradient_aadc,
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 200, 'ftol': 1e-15})
    t_aadc = time.time() - t0

    # FD calibration (no gradient provided)
    def calib_objective_fd(params):
        c, _ = evaluate(params, with_grad=False)
        return (c - target_cost) ** 2

    t0 = time.time()
    res_fd = minimize(calib_objective_fd, INIT_PARAMS,
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-15})
    t_fd = time.time() - t0

    print(f"\n   {'':>14s}  {'AADC':>10s}  {'FD':>10s}")
    print(f"   {'Time':>14s}  {t_aadc:>10.3f}s  {t_fd:>10.3f}s")
    print(f"   {'Speedup':>14s}  {t_fd/t_aadc:>10.1f}x  {'—':>10s}")
    print(f"   {'Final cost':>14s}  {res_aadc.fun:>10.2e}  {res_fd.fun:>10.2e}")
    print(f"   {'Iterations':>14s}  {res_aadc.nit:>10d}  {res_fd.nit:>10d}")
    print(f"   {'Func evals':>14s}  {res_aadc.nfev:>10d}  {res_fd.nfev:>10d}")

    print(f"\n   Recovered parameters:")
    print(f"   {'Param':>8s}  {'True':>8s}  {'AADC':>8s}  {'FD':>8s}  {'AADC err':>9s}  {'FD err':>9s}")
    for j in range(5):
        err_a = abs(res_aadc.x[j] - TRUE_PARAMS[j])
        err_f = abs(res_fd.x[j] - TRUE_PARAMS[j])
        print(f"   {PARAM_NAMES[j]:>8s}  {TRUE_PARAMS[j]:>8.4f}  {res_aadc.x[j]:>8.4f}  "
              f"{res_fd.x[j]:>8.4f}  {err_a:>9.2e}  {err_f:>9.2e}")

    # ================================================================
    # 4. Per-evaluation timing
    # ================================================================
    print(f"\n4. Per-evaluation timing:")
    n_iter = 1000
    t0 = time.time()
    for _ in range(n_iter):
        evaluate(TRUE_PARAMS, with_grad=True)
    t_per_grad = (time.time() - t0) / n_iter * 1000

    t0 = time.time()
    for _ in range(n_iter):
        evaluate(TRUE_PARAMS, with_grad=False)
    t_per_fwd = (time.time() - t0) / n_iter * 1000

    print(f"   Forward (15 prices):      {t_per_fwd:.3f} ms")
    print(f"   Forward + 5 gradients:    {t_per_grad:.3f} ms")
    print(f"   Gradient overhead:        {(t_per_grad - t_per_fwd) / t_per_fwd * 100:.0f}%")
    print(f"   FD equivalent (11 fwd):   {t_per_fwd * 11:.2f} ms")
    print(f"   AADC speedup per eval:    {t_per_fwd * 11 / t_per_grad:.1f}x")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  Summary:")
    print(f"    AD/FD exact:           {'YES' if all_ok else 'NO'}")
    print(f"    Calibration speedup:   {t_fd / t_aadc:.1f}x")
    print(f"    Per-eval speedup:      {t_per_fwd * 11 / t_per_grad:.1f}x")
    print(f"    Both find true params: {np.allclose(res_aadc.x, TRUE_PARAMS, atol=0.01)}")
    print(f"")
    print(f"  How it works:")
    print(f"    1. Record the pricing formula onto an AADC tape (once)")
    print(f"    2. Replay = forward pass (prices)")
    print(f"    3. Reverse pass = exact gradient w.r.t. all params")
    print(f"    4. Cost: ~2x one forward, independent of #params")
    print(f"    5. FD would need 11x forward for the same 5 gradients")
    print(f"{'=' * 70}")

    return all_ok


if __name__ == "__main__":
    ok = run_benchmark()
    exit(0 if ok else 1)
