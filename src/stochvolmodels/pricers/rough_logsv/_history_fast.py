"""Algebraically fused implementation of the audited full-history split."""

import numpy as np
from numba import njit


@njit(cache=False)
def simulate(normal, stops, s0, nu, rho, tilt, k0, d, dt, r, qstar, sbar, decay):
    n, paths = normal.shape
    nf = len(r)
    output = np.zeros((4, len(stops), paths))
    minimum = np.inf
    sqdt = np.sqrt(dt)
    A = decay * decay
    B = decay * r / k0
    D = decay * r
    co = k0 * d * dt / 2
    logco = np.log(co) if co > 0 else 0.0
    for path in range(paths):
        v = -qstar.copy()
        held = s0
        first = sbar + np.dot(D, v)
        I = 0.0
        J = 0.0
        target = 0
        for step in range(n):
            if held <= 0 or first <= 0 or not np.isfinite(held + first):
                raise ValueError("Nonpositive or nonfinite held/half-flow state")
            z = normal[step, path]
            I += held * held * dt
            J += held * sqdt * z
            shifted = z + tilt * rho * held * sqdt
            before_noise = first / (1 + co * first)
            ln = np.log(before_noise) + k0 * nu * sqdt * shifted
            if co > 0:
                lp = logco + ln
                after = (
                    np.exp(-logco - np.log1p(np.exp(-lp)))
                    if lp > 0
                    else np.exp(ln - np.log1p(np.exp(lp)))
                )
            else:
                after = np.exp(ln)
            delta = after - first
            held = sbar
            first = sbar
            for f in range(nf):
                v[f] = A[f] * v[f] + B[f] * delta
                held += r[f] * v[f]
                first += D[f] * v[f]
            minimum = min(minimum, held, after)
            if held <= 0 or not np.isfinite(held + I + J):
                raise ValueError("Invalid output; no floors, resets or deletions")
            if step + 1 == stops[target]:
                output[0, target, path] = I
                output[1, target, path] = J
                output[2, target, path] = held
                target += 1
    return output, minimum


def paths(prep, normal, days, grid):
    # Path-major contiguous access avoids strided access to a long Brownian tape.
    z = np.asfortranarray(normal, dtype=float)
    return simulate(
        z,
        np.array(days, dtype=np.int64) * grid,
        prep["s0"],
        prep["nu"],
        prep["rho"],
        prep["tilt"],
        prep["k0"],
        prep["d"],
        prep["dt"],
        prep["r"],
        prep["qstar"],
        prep["sigma_bar"],
        prep["decay"],
    )
