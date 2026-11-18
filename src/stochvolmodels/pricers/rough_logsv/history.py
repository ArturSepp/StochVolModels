"""Validated full-history solver and bounded conditional payoff interface.

Adapted from the November 2026 rough-logSV paper workflow. The finite kernel,
adapted price rule, Stratonovich correction and complete factor state are retained.
An invalid state raises: no volatility floors, state resets or path deletion.
"""

import numpy as np
from scipy.special import log_ndtr

from stochvolmodels.pricers.rough_logsv import _history, _history_fast
from stochvolmodels.pricers.rough_logsv.params import RoughKernel, RoughLogSvParams


def prepare_history(params: RoughLogSvParams, kernel: RoughKernel, dt: float, tilt=0.5):
    """Prepare exact linear memory flows for a fixed time step and sampling tilt.

    This solver requires a positive linear-flow spectrum. The restriction is
    numerical and is not a statement about nonlinear model well-posedness.
    Tilt zero is the ordinary-measure raw-forward run; tilt 1/2 prices options.
    """
    params.__post_init__()
    if params.H != kernel.H:
        raise ValueError("Parameter H and kernel target H differ")
    if not np.isfinite(dt) or dt <= 0 or tilt not in (0.0, 0.5):
        raise ValueError("Positive dt and tilt zero or one half are required")
    if params.vartheta == 0:
        raise ValueError(
            "History splitting requires nonzero noise; use constant-volatility pricing"
        )
    try:
        return _history.prepare(
            _history.model(params), np.array(kernel.nodes), np.array(kernel.weights), dt, tilt
        )
    except AssertionError as exc:
        raise ValueError("Kernel/model outside the positive-spectrum solver domain") from exc


def simulate_history(prepared, normal, observation_steps):
    """Return I, J, volatility and reserved-zero diagnostics at all requested steps.

    A single path traverses every maturity, retaining every factor. normal
    contains standard Gaussian increments, with shape (steps, paths).
    I and J use the volatility held at the start of each step.
    """
    z = np.asarray(normal, dtype=float)
    stops = np.asarray(observation_steps)
    if (
        z.ndim != 2
        or min(z.shape) < 1
        or not np.isfinite(z).all()
        or stops.ndim != 1
        or len(stops) == 0
        or not np.isfinite(stops).all()
        or np.any(stops != np.floor(stops))
        or stops[0] < 1
        or np.any(np.diff(stops) <= 0)
        or stops[-1] != z.shape[0]
    ):
        raise ValueError(
            "Finite Gaussian tape and increasing integer stops ending at its length required"
        )
    p = prepared
    return _history_fast.simulate(
        np.asfortranarray(z),
        stops.astype(np.int64),
        p["s0"],
        p["nu"],
        p["rho"],
        p["tilt"],
        p["k0"],
        p["d"],
        p["dt"],
        p["r"],
        p["qstar"],
        p["sigma_bar"],
        p["decay"],
    )


def conditional_payoffs(integral, jump, rho, strikes_over_forward, is_call):
    """Bounded half-tilted conditional payoffs, in units of the quoted forward.

    Inputs broadcast over quotes and paths. This is the paper's half-measure
    complement formula; it requires the sufficient martingale condition at the
    pricer boundary. It does not divide by an empirical forward estimate.
    """
    I, J, k, call = np.broadcast_arrays(
        np.asarray(integral, float),
        np.asarray(jump, float),
        np.asarray(strikes_over_forward, float),
        np.asarray(is_call, bool),
    )
    if (
        not np.isfinite(I).all()
        or not np.isfinite(J).all()
        or not np.isfinite(k).all()
        or np.any(I < 0)
        or np.any(k <= 0)
        or not np.isfinite(rho)
        or abs(rho) > 1
    ):
        raise ValueError("Invalid integrals, correlation or normalized strikes")
    v = (1 - rho * rho) * I
    z = rho * J - np.log(k)
    log_complement = np.array(-I / 8 - np.abs(z) / 2)
    mask = v > 0
    root = np.sqrt(v[mask])
    first = -z[mask] / 2 + v[mask] / 8 + log_ndtr((z[mask] - v[mask] / 2) / root)
    second = z[mask] / 2 + v[mask] / 8 + log_ndtr((-z[mask] - v[mask] / 2) / root)
    log_complement[mask] = -I[mask] / 8 + np.logaddexp(first, second)
    complement = np.sqrt(k) * np.exp(log_complement)
    if not np.isfinite(complement).all() or np.any(complement > np.sqrt(k) + 1e-12):
        raise FloatingPointError("Invalid bounded conditional payoff")
    return np.where(call, 1.0, k) - complement


def raw_forward(integral, jump, rho):
    """Ordinary-measure conditional forward, with no empirical normalization."""
    I, J = np.asarray(integral, float), np.asarray(jump, float)
    if (
        not np.isfinite(I).all()
        or not np.isfinite(J).all()
        or np.any(I < 0)
        or not np.isfinite(rho)
        or abs(rho) > 1
    ):
        raise ValueError("Finite nonnegative variance and correlation in [-1,1] required")
    value = np.exp(rho * J - 0.5 * rho * rho * I)
    if not np.isfinite(value).all():
        raise FloatingPointError("Nonfinite raw conditional forward")
    return value


def exact_linear_discrete_moments(prepared, observation_steps):
    """Exact discrete first/second moments and adapted integrated variance.

    Applies only to the linear-drift, ordinary-measure restriction. This is a
    deterministic moment recursion, independent of stochastic path sampling.
    """
    p = prepared
    if p["tilt"] != 0 or p["d"] != 0:
        raise ValueError("Exact discrete moments require linear drift and ordinary measure")
    stops = np.asarray(observation_steps)
    if (
        stops.ndim != 1
        or not len(stops)
        or not np.isfinite(stops).all()
        or np.any(stops != np.floor(stops))
        or stops[0] <= 0
        or np.any(np.diff(stops) <= 0)
    ):
        raise ValueError("Increasing positive integer observation steps required")
    return _history.linear_moments(
        stops.astype(np.int64),
        p["s0"],
        p["nu"],
        p["k0"],
        p["dt"],
        p["r"],
        p["qstar"],
        p["decay"],
        p["reward"],
    )
