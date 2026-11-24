"""Rebuild the positive 97-factor kernel used by the BTC and ETH rough fits in Table 5."""

import argparse
from enum import StrEnum
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma, gammaincc, roots_jacobi, roots_legendre


SAVED = Path(__file__).resolve().parent / "inputs/kernels/H0.3_s512_n8.json"
SAVED_SHA256 = "fe1fca38b667f536804d0fe19e7da79ee2ab554928035f357590faebe2709331"


class Locals(StrEnum):
    REBUILD = "rebuild"


def table5_kernel():
    """Discretise the two-mass regularisation at H=.30 with eight nodes per interval."""
    hurst, order = 0.30, 8
    alpha, power = hurst + 0.5, 0.5 - hurst
    delta = 1.0 / (365 * 512)
    normalizer = gamma(alpha)

    def square_mass_error(log_rate):
        rate = np.exp(log_rate)
        cross = np.exp(rate + (1 - alpha) * log_rate) * normalizer * gammaincc(alpha, rate) / alpha
        return -1 / (2 * hurst) + 2 * cross + rate / (2 * alpha**2)

    rate = np.exp(brentq(square_mass_error, -700.0, np.log(100.0), xtol=1e-13))
    amplitude = rate / alpha
    constant = 1 / (gamma(power) * normalizer)
    cutoff = 40 / delta
    left, right = 0.0, 1 / (91 / 365)
    nodes, weights = [], []
    while left < cutoff:
        if left == 0:
            z, w = roots_jacobi(order, 0.0, power - 1)
            x = right * (z + 1) / 2
            w = w * (right / 2) ** power
        else:
            z, w = roots_legendre(order)
            x = left + (right - left) * (z + 1) / 2
            w = w * ((right - left) / 2) * x ** (power - 1)
        nodes.extend(x)
        weights.extend(constant * w * np.exp(-delta * x))
        left, right = right, min(4 * right, cutoff)

    nodes.append(rate / delta)
    weights.append(amplitude * delta ** (-power) / normalizer)
    return np.asarray(nodes), np.asarray(weights)


def run_local(local=Locals.REBUILD, output=None):
    """Compare each rebuilt coefficient to its saved counterpart in the original order."""
    if local != Locals.REBUILD:
        raise ValueError(local)
    raw = SAVED.read_bytes()
    fingerprint = sha256(raw).hexdigest()
    if fingerprint != SAVED_SHA256:
        raise ValueError(f"Unexpected SHA-256 for {SAVED.name}: {fingerprint}")
    saved = json.loads(raw)
    nodes, weights = table5_kernel()
    errors = {}
    for name, rebuilt in (("nodes", nodes), ("weights", weights)):
        target = np.asarray(saved[name])
        if rebuilt.shape != (97,) or target.shape != rebuilt.shape or not np.all(target > 0):
            raise ValueError(f"Expected 97 positive {name}")
        errors[name] = float(np.max(np.abs(rebuilt - target) / target))
        np.testing.assert_allclose(rebuilt, target, rtol=1e-12, atol=0)
    check = dict(saved_sha256=fingerprint, factors=len(nodes), max_relative_difference=errors)
    if output is not None:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(dict(nodes=nodes.tolist(), weights=weights.tolist(), comparison=check), stream, indent=2)
            stream.write("\n")
    print(json.dumps(check, indent=2))
    return check


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Locals, choices=list(Locals), default=Locals.REBUILD)
    parser.add_argument("--output", type=Path, help="Optional new JSON file for the rebuilt coefficients")
    args = parser.parse_args()
    run_local(args.case, args.output)
