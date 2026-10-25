"""Private experimental positive kernel fits with hard geometric constraints.

The existing public moment-fit rule is unchanged. This research helper maps a
bounded parameter vector into positive nodes/weights and unused weight-cap slack,
then fits the existing linear-model moment targets by local least squares. A
solver success flag does not certify kernel error, nonlinear prices or time-step
accuracy. Independent moment and price qualification remains necessary.
"""
from time import perf_counter
import warnings

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, softmax

from stochvolmodels.pricers.rough_logsv.kernel_fit import (
    DEFAULT_FIT_PARAMS, MATURITY_FRACTIONS, MomentKey, fractional_moment_targets,
    markov_moment_targets,
)

_PARAMETER_BOUND = 20.0
_CAP_RESERVE = 16.0 * np.finfo(float).eps


def _decode_kernel(parameters, n_nodes, node_min, node_cap, weight_floor, weight_cap):
    """Map finite bounded coordinates into the strict interior of the caps."""
    z = np.asarray(parameters, dtype=float)
    if z.shape != (2 * n_nodes,) or not np.all(np.isfinite(z)):
        raise ValueError('kernel coordinates must be a finite vector of length 2*N')
    if np.any(np.abs(z) > _PARAMETER_BOUND):
        raise ValueError('kernel coordinates lie outside the bounded search domain')
    log_nodes = np.log(node_min) + np.log(node_cap / node_min) * expit(z[:n_nodes])
    nodes = np.exp(log_nodes)
    probabilities = softmax(np.r_[z[n_nodes:], 0.0])
    available = (weight_cap - n_nodes * weight_floor) * (1.0 - _CAP_RESERVE)
    weights = weight_floor + available * probabilities[:-1]
    return nodes, weights


def _encode_kernel(nodes, weights, node_min, node_cap, weight_floor, weight_cap):
    """Encode an already feasible interior seed; never repair an invalid seed."""
    x, w = np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float)
    if x.shape != w.shape or x.ndim != 1 or not x.size:
        raise ValueError('seed nodes and weights must have equal nonempty 1D shapes')
    if (not np.all(np.isfinite(np.r_[x, w])) or np.any(x <= node_min)
            or np.any(x >= node_cap) or np.any(w <= weight_floor)):
        raise ValueError('seed must lie strictly inside the node and weight floors/caps')
    p = np.log(x / node_min) / np.log(node_cap / node_min)
    node_coordinates = np.log(p / (1.0 - p))
    available = (weight_cap - len(w) * weight_floor) * (1.0 - _CAP_RESERVE)
    probabilities = (w - weight_floor) / available
    slack = 1.0 - probabilities.sum()
    if slack <= 0.0:
        raise ValueError('seed must leave strictly positive total-weight slack')
    result = np.r_[node_coordinates, np.log(probabilities / slack)]
    if np.any(np.abs(result) >= _PARAMETER_BOUND):
        raise ValueError('seed coordinates must lie inside the numerical search bounds')
    return result


def _validate_achieved_kernel(nodes, weights, node_min, node_cap, weight_floor, weight_cap):
    """Reject any actual cap violation; no feasibility tolerance or final clipping."""
    x, w = np.asarray(nodes), np.asarray(weights)
    if (x.ndim != 1 or x.shape != w.shape or not x.size
            or not np.all(np.isfinite(np.r_[x, w]))
            or np.any(x < node_min) or np.any(x > node_cap)
            or np.any(w < weight_floor) or np.any(w <= 0.0)
            or np.sum(w) > weight_cap):
        raise ValueError('achieved kernel violates a hard node/positive-weight/total-weight constraint')


def _spectral_seed(hurst, n_nodes, node_min, node_cap, weight_floor, weight_cap):
    """Deterministic feasible seed shaped like the fractional Laplace measure."""
    lower, upper = max(node_min * 20.0, 0.03), node_cap * 0.8
    if node_min < lower < upper < node_cap:
        # Preserve the previously executed seed on the research domain.
        nodes = np.geomspace(lower, upper, n_nodes)
    else:
        fractions = np.array([0.5]) if n_nodes == 1 else np.linspace(.15, .85, n_nodes)
        nodes = node_min * np.exp(np.log(node_cap / node_min) * fractions)
    proportions = nodes ** (0.5 - hurst)
    proportions /= proportions.sum()
    weights = weight_floor + 0.65 * (weight_cap - n_nodes * weight_floor) * proportions
    return nodes, weights


def fit_positive_kernel(hurst, n_nodes, weight_cap, node_cap=360.0,
                        maturities=MATURITY_FRACTIONS, fit_params=DEFAULT_FIT_PARAMS,
                        targets=None, initial_kernel=None, max_nfev=80,
                        weight_floor=1e-3, node_min=None, tolerance=1e-7):
    """Fit a strictly positive kernel without changing any public pricing rule.

    Parameters
    ----------
    hurst : float
        Fractional-kernel exponent in (0, 0.5]; the endpoint returns x=0, w=1.
    n_nodes : int
        Requested number of factors. Effective dimension at H=0.5 is one.
    weight_cap, node_cap : float
        Hard upper bounds on sum(weights) and max(nodes). Nodes are in 1/years.
    targets : sequence of dict, optional
        One existing MomentKey target dictionary per training parameter set.
    initial_kernel : pair of arrays, optional
        Already feasible nodes/weights, never repaired or silently rescaled.
    max_nfev : int
        Least-squares function-evaluation limit; numerical-Jacobian calls are
        counted separately in the returned actual_objective_calls field.

    Returns
    -------
    dict
        Nodes/weights, actual solver termination and constraint diagnostics.
        ``success`` means optimizer success with a finite objective, not an
        independent accuracy acceptance. Feasible budget-exhausted iterates are
        retained with success=False for diagnosis, never relabeled as converged.
    """
    started = perf_counter()
    if not np.isscalar(hurst) or not np.isfinite(hurst) or not 0.0 < hurst <= 0.5:
        raise ValueError('hurst must be finite and in (0, 0.5]')
    if isinstance(n_nodes, (bool, np.bool_)) or not isinstance(n_nodes, (int, np.integer)) or n_nodes < 1:
        raise ValueError('n_nodes must be a positive integer')
    times = np.asarray(maturities, dtype=float)
    if (times.ndim != 1 or not times.size or not np.all(np.isfinite(times))
            or np.any(times <= 0.0)):
        raise ValueError('maturities must be a finite positive vector')
    lower_node = 1e-3 / np.max(times) if node_min is None else node_min
    if (not np.all(np.isfinite([weight_cap, node_cap, weight_floor, lower_node, tolerance]))
            or min(weight_cap, node_cap, weight_floor, lower_node, tolerance) <= 0.0):
        raise ValueError('caps, floors and tolerance must be finite and positive')
    if isinstance(max_nfev, (bool, np.bool_)) or not isinstance(max_nfev, (int, np.integer)) or max_nfev < 1:
        raise ValueError('max_nfev must be a positive integer')
    if hurst == 0.5:
        if weight_cap < 1.0 or weight_floor > 1.0:
            raise ValueError('classical constant kernel requires weight_floor <= 1 <= weight_cap')
        return dict(H=hurst, N=n_nodes, effective_N=1, nodes=[0.0], weights=[1.0],
                    success=True, status='exact_classical', solver_status=None,
                    message='Exact constant kernel; no fit performed', objective=0.0,
                    weight_sum=1.0, max_node=0.0, w_bar_max=float(weight_cap),
                    x_max=float(node_cap), node_min=0.0, weight_floor=float(weight_floor),
                    feasible=True, actual_objective_calls=0, nfev=0, njev=0,
                    elapsed_seconds=perf_counter() - started)
    if node_cap <= lower_node or weight_cap <= n_nodes * weight_floor:
        raise ValueError('caps must leave a strictly positive node interval and weight slack')
    parameters = [tuple(p) for p in fit_params]
    if not parameters:
        raise ValueError('at least one training parameter set is required')
    for p in parameters:
        if len(p) != 4 or not np.all(np.isfinite(p)) or min(p) < 0.0:
            raise ValueError('training parameters must be finite nonnegative quadruples')
    if targets is None:
        targets = [fractional_moment_targets(hurst, *p, maturities=times) for p in parameters]
    if len(targets) != len(parameters):
        raise ValueError('one target dictionary is required per training parameter set')
    keys = tuple(MomentKey)
    target_values = []
    for target in targets:
        values = np.array([target[key] for key in keys], dtype=float)
        if values.shape != (len(keys), len(times)) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError('moment targets must be positive, finite and maturity-aligned')
        target_values.append(values)
    seed = (_spectral_seed(hurst, n_nodes, lower_node, node_cap, weight_floor, weight_cap)
            if initial_kernel is None else initial_kernel)
    if len(seed[0]) != n_nodes:
        raise ValueError('initial kernel dimension differs from requested n_nodes')
    initial = _encode_kernel(*seed, lower_node, node_cap, weight_floor, weight_cap)
    calls = 0
    numerical_failures = 0

    def residuals(z):
        nonlocal calls, numerical_failures
        calls += 1
        nodes, weights = _decode_kernel(z, n_nodes, lower_node, node_cap, weight_floor, weight_cap)
        residual = []
        try:
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                for p, target in zip(parameters, target_values):
                    got = markov_moment_targets(nodes, weights, *p, maturities=times)
                    residual.extend((np.array([got[key] for key in keys]) / target - 1.0).ravel())
            residual = np.asarray(residual)
            if not np.all(np.isfinite(residual)):
                raise FloatingPointError('nonfinite trial moments')
            return residual
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError):
            numerical_failures += 1
            return np.full(len(parameters) * len(keys) * len(times), 1e6)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = least_squares(residuals, initial, bounds=(-_PARAMETER_BOUND, _PARAMETER_BOUND),
                               max_nfev=max_nfev, ftol=tolerance, xtol=tolerance,
                               gtol=tolerance, x_scale='jac', method='trf')
    nodes, weights = _decode_kernel(result.x, n_nodes, lower_node, node_cap, weight_floor, weight_cap)
    _validate_achieved_kernel(nodes, weights, lower_node, node_cap, weight_floor, weight_cap)
    order = np.argsort(nodes)
    objective = float(np.dot(result.fun, result.fun))
    finite_fit = bool(np.isfinite(objective) and np.max(np.abs(result.fun)) < 1e6)
    return dict(H=float(hurst), N=int(n_nodes), effective_N=int(n_nodes),
                nodes=nodes[order].tolist(), weights=weights[order].tolist(),
                success=bool(result.success and finite_fit),
                status=('converged' if result.success and finite_fit else
                        'budget_exhausted_feasible' if result.status == 0 and finite_fit else
                        'optimizer_failed_feasible'),
                solver_status=int(result.status), message=str(result.message),
                objective=objective, max_training_relative_error=float(np.max(np.abs(result.fun))),
                weight_sum=float(weights.sum()), max_node=float(nodes.max()),
                w_bar_max=float(weight_cap), x_max=float(node_cap), node_min=float(lower_node),
                weight_floor=float(weight_floor), feasible=True,
                cap_weight_fraction=float(weights.sum() / weight_cap),
                cap_node_fraction=float(nodes.max() / node_cap),
                min_node=float(nodes.min()), min_weight=float(weights.min()),
                actual_objective_calls=calls, numerical_failure_calls=numerical_failures,
                nfev=int(result.nfev), njev=int(result.njev),
                optimality=float(result.optimality), parameters=result.x.tolist(),
                initial_nodes=np.asarray(seed[0]).tolist(), initial_weights=np.asarray(seed[1]).tolist(),
                warnings=[str(w.message) for w in caught],
                elapsed_seconds=perf_counter() - started)
