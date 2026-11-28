"""Reproduce the daily-close accumulator anchors by Gaussian transition quadrature.

Use a fixed log-spot grid ending exactly on the knock-out. At each fixing,
integrate only over the surviving half-line. This calculation uses neither the
SVM finite-difference solver nor its log-forward barrier cell projection.
"""

import numpy as np
from scipy.optimize import brentq


def fair_strikes(nodes: int, *, buyer: bool, carry: float) -> tuple[float, float]:
    """Return par strikes under daily payments and payments every 21st close."""
    count, sigma, rate = 252, 0.20, 0.03
    barrier = np.log(1.05 if buyer else 0.95)
    lower, upper = (-1.2, barrier) if buyer else (barrier, 1.2)
    x = np.linspace(lower, upper, nodes)
    dx = x[1] - x[0]
    weights = np.full(nodes, dx)
    weights[[0, -1]] *= 0.5
    variance = sigma**2 / count
    drift = (carry - 0.5 * sigma**2) / count

    def transition(displacement):
        return np.exp(-0.5 * (displacement - drift) ** 2 / variance) / np.sqrt(2 * np.pi * variance)

    radius = int(np.ceil(10 * np.sqrt(variance) / dx))
    kernel = transition(dx * np.arange(-radius, radius + 1))
    alive = transition(x)
    totals = np.zeros((2, nodes))
    for fixing in range(1, count + 1):
        if fixing > 1:
            alive = np.convolve(alive * weights, kernel, mode="same")
        monthly = 21 * ((fixing + 20) // 21)
        totals[0] += np.exp(-rate * fixing / count) * alive
        totals[1] += np.exp(-rate * monthly / count) * alive

    spot = np.exp(x)
    sign = 1 if buyer else -1

    def root(measure):
        def value(k):
            signed = sign * (spot - k)
            return float(measure @ (signed + np.minimum(signed, 0)))

        return brentq(value, 0.5, 2.0, xtol=1e-13)

    daily, monthly = totals * weights / count
    return root(daily), root(monthly)


def main() -> None:
    """Print the twelve reference strikes and check their spatial refinement in spot bp."""
    for buyer in (True, False):
        for carry in (0.0, 0.03, -0.02):
            coarse = fair_strikes(3201, buyer=buyer, carry=carry)
            fine = fair_strikes(6401, buyer=buyer, carry=carry)
            delta = 1e4 * (np.asarray(coarse) - fine)
            assert np.max(np.abs(delta)) < 0.009
            name = "accumulator" if buyer else "decumulator"
            print(
                f"{name:12s} carry={carry:+.0%} daily={fine[0]:.9f} monthly={fine[1]:.9f} "
                f"coarse-fine bp=({delta[0]:+.4f}, {delta[1]:+.4f})"
            )


if __name__ == "__main__":
    main()
