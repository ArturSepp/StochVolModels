"""Forward sensitivities of the classical vanilla log-return transform for calibration.

PARAMS4 fixes both mean-reversion rates. PARAMS5 fits kappa1 with kappa2=kappa1/theta.
Both use the unit calendar backbone and propagate the coefficient tangents on the
pricing solver's accepted RK45 steps. Initial volatility enters at contraction.
"""

import numpy as np
from numba import njit
from scipy.integrate import RK45

from stochvolmodels.models.logsv import affine_expansion as afe
from stochvolmodels.models.logsv.params import LogSvParams


def _coefficient_stencil(params: LogSvParams, fit_kappa1: bool) -> tuple:
    """Primitive coefficient inputs and exact polarization directions in optimizer order."""
    point = np.array([params.theta, params.kappa1, params.kappa2, params.beta, params.volvol])
    free = [0, 1, 3, 4] if fit_kappa1 else [0, 3, 4]
    directions = np.eye(point.size)[free]
    if fit_kappa1:
        # On kappa2=kappa1/theta, both free coordinates move kappa2.
        directions[0, 2] = -params.kappa2 / params.theta
        directions[1, 2] = 1. / params.theta
    steps = .125 * np.maximum(1., np.abs(point[free]))
    deltas = steps[:, None] * directions
    return point, point + deltas, point - deltas, 2. * steps


def _tensors(stencil, phi: complex) -> tuple:
    """Assemble M, L, H and their total derivatives along each free direction."""
    point, upper, lower, widths = stencil

    def assemble(p):
        return afe.func_a_ode_quadratic_terms(
            p[0], p[1], p[2], p[3], p[4], phi, 0j,
            True, afe.ExpansionOrder.SECOND, 1.,
        )

    tensors = assemble(point)
    partials = tuple(np.empty((len(widths), *value.shape), dtype=complex) for value in tensors)
    # Along each straight tangent in primitive coordinates, every tensor entry
    # is at most quadratic (including theta*kappa2). Polarization is exact.
    for j, (plus, minus, width) in enumerate(zip(upper, lower, widths, strict=True)):
        for derivative, hi, lo in zip(partials, assemble(plus), assemble(minus), strict=True):
            derivative[j] = (hi - lo) / width
    return (*tensors, *partials)


@njit(cache=False)
def _row_dot(row, a):
    # Five entries. A scalar loop avoids a BLAS dispatch per quadratic-form term.
    value = 0j
    for k in range(a.size):
        value += row[k] * a[k]
    return value


@njit(cache=False)
def _stage(state, m, linear, source, dm, dl, dh):
    """Evaluate F and F_A S + F_p for all tangent columns at one Runge–Kutta stage."""
    a = state[0]
    n = a.size
    rhs = np.empty_like(state)
    for k in range(n):
        rhs[0, k] = source[k]
        for j in range(dh.shape[0]):
            rhs[j + 1, k] = dh[j, k]
        for i in range(n):
            ma = _row_dot(m[k, i], a)
            rhs[0, k] += a[i] * (ma + linear[k, i])
            jac = 2. * ma + linear[k, i]  # the quadratic-form matrices are symmetric
            for j in range(dh.shape[0]):
                rhs[j + 1, k] += jac * state[j + 1, i] + a[i] * (_row_dot(dm[j, k, i], a) + dl[j, k, i])
    return rhs


@njit(cache=False)
def _propagate(times, primal, slopes, tableau, weights, m, linear, source, dm, dl, dh):
    """Replay the accepted RK45 stages and differentiate the coefficient flow."""
    start = np.empty((1 + slopes.shape[0], primal.shape[1]), dtype=np.complex128)
    stages = np.empty((len(weights), *start.shape), dtype=np.complex128)
    start[1:] = slopes
    for t in range(len(times) - 1):
        dt = times[t+1] - times[t]
        start[0] = primal[t]
        for s in range(len(weights)):
            state = start.copy()
            for earlier in range(s):
                state += (dt * tableau[s, earlier]) * stages[earlier]
            stages[s] = _stage(state, m, linear, source, dm, dl, dh)
        for s in range(len(weights)):
            start[1:] += (dt * weights[s]) * stages[s, 1:]
    return start[1:].copy()


def calibration_transform_jacobian(params: LogSvParams, ttms: np.ndarray, phi: np.ndarray,
                                   *, fit_kappa1: bool):
    """Return log M and dM/d(sigma0, theta, [kappa1], beta, volvol) for each expiry."""
    if params.vol_backbone is not None:
        raise ValueError("calibration coefficient sensitivities require the fixed unit backbone")
    stencil = _coefficient_stencil(params, fit_kappa1)
    n_tangents = len(stencil[1])
    basis = afe._state_basis(params.sigma0 - params.theta, 5)
    basis_prime = np.arange(1, 5) * basis[:-1]
    intervals = np.diff(np.r_[0., ttms])
    logs = np.empty((len(ttms), phi.size), dtype=complex)
    derivatives = np.empty((*logs.shape, n_tangents + 1), dtype=complex)
    for col, node in enumerate(phi):
        tensors = _tensors(stencil, node)
        a = np.zeros(5, dtype=complex)
        slopes = np.zeros((n_tangents, 5), dtype=complex)
        for i, dt in enumerate(intervals):
            solution = afe.solve_ode_for_a(dt, params.theta, params.kappa1, params.kappa2,
                                           params.beta, params.volvol, node, 0j, a_t0=a,
                                           expansion_order=afe.ExpansionOrder.SECOND)
            slopes = _propagate(solution.t, np.ascontiguousarray(solution.y.T), slopes,
                                RK45.A, RK45.B, *tensors)
            a = solution.y[:, -1]
            logs[i, col] = a @ basis
            initial_vol = a[1:] @ basis_prime
            row = derivatives[i, col]
            row[0] = initial_vol
            row[1:] = slopes @ basis
            row[1] -= initial_vol  # theta also moves the expansion centre
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                row *= np.exp(logs[i, col])
            if not np.isfinite(row).all():
                raise FloatingPointError(f"nonfinite calibration derivative at expiry {ttms[i]:g}, phi={node}")
    return zip(logs, derivatives, strict=True)
