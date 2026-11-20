"""Experimental positive full-tensor generator; repository-only, no production defaults.

Moment-based wide stencils are motivated by Reisinger (2016), section 2,
https://arxiv.org/abs/1605.06348. Positive directional interpolation is discussed
by Debrabant & Jakobsen (2013), section 3, https://arxiv.org/abs/0910.1046.
The construction below is an independent research prototype, not an implementation
of a theorem from either paper. Its boundary closure and fallbacks need qualification.

Split the covariance into a correlated rank-one part and residual axial variances.
Interpolate each paired ray with positive bilinear weights; subtract its actual
second moments from the axial budgets. Widen the ray until both budgets fit, or
record explicit diagonal diffusion inflation at the configured radius limit.
Interior cross covariance is retained, including both correlation signs.

At an axis endpoint freeze its normal dynamics and retain tangential dynamics,
except at y=0 with exactly zero normal diffusion and covariance: retain inward
drift b using the positive rate b/h to the first interior node. This preserves
the first moment and adds a recorded second moment b*h, vanishing with spacing.
The optional upper-volatility reflection returns mass to the adjacent node at the same x.
For last spacing h, physical normal variance a and drift b its rate is a/h²+max(-b,0)/h.
There is no exterior transition. Tangential dynamics continue along the face, mixed jumps
from the face are suppressed, and at an x corner only the vertical return remains.
By default the upper face retains its absorbing-normal closure. No finite-state positive
generator can retain nonzero variance and a zero-drift coordinate at its extreme.
Optional exponential fitting preserves exp(x) for money-market log-forward models,
with measured log-drift error. All other drift uses centered weights when positive,
otherwise upwind with measured added variance. No resulting density is repaired.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from stochvolmodels.pde_solvers import BoundaryKind
from validation.pde_unsplit import Unsplit2DReference, UnsplitConfig


@dataclass(frozen=True)
class MonotoneConfig:
    """Bound the directional search; optionally reject any diffusion-budget inflation."""

    radii: tuple[int, ...] = (1, 2, 4, 8)
    fit_log_forward: bool = True
    reject_inflation: bool = False
    retain_zero_vol_drift: bool = True
    reflect_upper_vol: bool = False

    def __post_init__(self):
        if (not isinstance(self.radii, tuple) or not self.radii
                or any(isinstance(r, bool) or not isinstance(r, int) or r < 1 for r in self.radii)
                or list(self.radii) != sorted(set(self.radii)) or self.radii[-1] > 32):
            raise ValueError("radii must be increasing distinct integers between 1 and 32")
        if (not isinstance(self.fit_log_forward, bool)
                or not isinstance(self.reject_inflation, bool)
                or not isinstance(self.retain_zero_vol_drift, bool)
                or not isinstance(self.reflect_upper_vol, bool)):
            raise TypeError(
                "fit_log_forward, reject_inflation, retain_zero_vol_drift and "
                "reflect_upper_vol must be bools")


def _weights(nodes, point):
    """Positive interpolation at a point already inside the closed domain."""
    right = int(np.searchsorted(nodes, point, side="right"))
    right = min(max(right, 1), nodes.size - 1)
    left = right - 1
    weight = (point - nodes[left]) / (nodes[right] - nodes[left])
    if weight < -1e-12 or weight > 1 + 1e-12:
        raise ValueError("ray interpolation outside domain")
    weight = min(max(float(weight), 0.), 1.)  # Roundoff at an endpoint, never density clipping.
    return ((left, 1 - weight), (right, weight))


def _ray(x, y, i, j, dx, dy, rho, length):
    """Paired unequal-length ray: zero first moment and exact cross covariance."""
    point = np.array([x[i], y[j]])
    direction = np.array([dx, dy])
    lower, upper = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    limits = []
    for sign in (-1, 1):
        d = sign * direction
        distance = np.where(d > 0, upper - point, point - lower)
        limits.append(min(length, float(np.min(distance / np.abs(d)))))
    minus, plus = limits
    if min(limits) <= 0:
        raise ValueError("a correlated ray requires an interior point")
    rates = {}
    for sign, h in ((-1, minus), (1, plus)):
        endpoint = np.minimum(np.maximum(point + sign*h*direction, lower), upper)
        rate = rho / (h * (minus + plus))
        for a, wa in _weights(x, endpoint[0]):
            for b, wb in _weights(y, endpoint[1]):
                if (a, b) != (i, j) and wa*wb > 0:
                    rates[a, b] = rates.get((a, b), 0.) + rate*wa*wb
    vx = sum(q*(x[a]-x[i])**2 for (a, b), q in rates.items())
    vy = sum(q*(y[b]-y[j])**2 for (a, b), q in rates.items())
    return rates, vx, vy, min(limits) < length


def _axial(variance, drift, hm, hp):
    """Nonuniform centered first/second moments, with explicit positive drift fallback."""
    qm = (variance - drift*hp) / (hm*(hm+hp))
    qp = (variance + drift*hm) / (hp*(hm+hp))
    fallback = min(qm, qp) < 0
    if fallback:
        qm = variance / (hm*(hm+hp)) + max(-drift, 0.) / hm
        qp = variance / (hp*(hm+hp)) + max(drift, 0.) / hp
    return qm, qp, fallback


@dataclass
class Monotone2DReference(Unsplit2DReference):
    """Nonnegative off-diagonal backward generator and its exact quadrature adjoint."""

    moment_errors: dict = field(default_factory=dict)

    @classmethod
    def from_split(cls, operator, config=None, *, spatial=None):
        """Build a bounded positive generator; expose every consistency modification."""
        config = UnsplitConfig() if config is None else config
        spatial = MonotoneConfig() if spatial is None else spatial
        if not isinstance(config, UnsplitConfig) or not isinstance(spatial, MonotoneConfig):
            raise TypeError("expected UnsplitConfig and MonotoneConfig")
        grid, coeff = operator.grid, operator.coefficients
        nx, ny = grid.shape
        n = nx * ny
        if n > config.max_nodes:
            raise ValueError(f"grid has {n} nodes, exceeding max_nodes={config.max_nodes}")
        if any(c.kind is not BoundaryKind.NATURAL for pair in
               (operator.boundaries.x, operator.boundaries.y) for c in (pair.left, pair.right)):
            raise ValueError("research closure supports natural boundaries only")
        if np.any(coeff.discount < 0):
            raise ValueError("positive BE guarantee requires nonnegative discount")
        if spatial.fit_log_forward and not np.allclose(
                coeff.drift_x, -coeff.variance_x, rtol=1e-12, atol=1e-14):
            raise ValueError("exponential fitting requires drift_x = -variance_x")
        x, y = grid.x.nodes, grid.y.nodes
        rows, columns, entries = [], [], []
        errors = {k: np.zeros(grid.shape) for k in
                  ("drift_x", "drift_y", "variance_x", "variance_y", "covariance_xy")}
        inflation = np.zeros(grid.shape, dtype=bool)
        radius_used = np.zeros(grid.shape, dtype=int)
        truncated = 0
        x_fallback = y_fallback = 0
        zero_vol_nodes = 0
        zero_vol_added_variance = np.zeros(grid.shape)
        max_covariance_error = 0.
        for i in range(nx):
            for j in range(ny):
                vx, vy = 2*coeff.variance_x[i, j], 2*coeff.variance_y[i, j]
                bx, by, c = coeff.drift_x[i, j], coeff.drift_y[i, j], coeff.covariance_xy[i, j]
                interior_x, interior_y = 0 < i < nx-1, 0 < j < ny-1
                zero_vol = (spatial.retain_zero_vol_drift and j == 0 and y[0] == 0.
                            and vy == 0. and c == 0.)
                if zero_vol and by < 0:
                    raise ValueError("outward drift at degenerate zero-volatility boundary")
                # Keep the physical entrance drift at the degenerate zero-volatility boundary.
                if not interior_x:
                    vx = bx = c = 0.
                if not interior_y:
                    vy = c = 0.
                    if not zero_vol:
                        by = 0.
                q = {}
                used_x = used_y = 0.
                if c != 0:
                    dx, dy = np.sqrt(vx), np.copysign(np.sqrt(vy), c)
                    rho = min(abs(c) / np.sqrt(vx*vy), 1.)
                    unit = max(max(x[i]-x[i-1], x[i+1]-x[i])/dx,
                               max(y[j]-y[j-1], y[j+1]-y[j])/abs(dy))
                    for radius in spatial.radii:
                        q, used_x, used_y, shortened = _ray(
                            x, y, i, j, dx, dy, rho, radius*unit)
                        if used_x <= vx*(1+1e-12) and used_y <= vy*(1+1e-12):
                            break
                    radius_used[i, j] = radius
                    truncated += int(shortened)
                    inflation[i, j] = used_x > vx*(1+1e-12) or used_y > vy*(1+1e-12)
                if inflation[i, j] and spatial.reject_inflation:
                    raise ValueError(f"diffusion budget needs inflation at node ({i}, {j})")
                if interior_x:
                    hm, hp = x[i]-x[i-1], x[i+1]-x[i]
                    remainder = max(vx-used_x, 0.)
                    if spatial.fit_log_forward:
                        em, ep = -np.expm1(-hm), np.expm1(hp)
                        convex = sum(rate*np.expm1(x[a]-x[i]) for (a, b), rate in q.items())
                        if convex < -1e-12 * max(vx, 1.):
                            raise ValueError("correlated ray lost convexity")
                        convex = max(convex, 0.)
                        qp = (remainder - hm*hm*convex/em) / (hp*hp + hm*hm*ep/em)
                        if qp >= 0:
                            qm = (convex + qp*ep) / em
                        else:
                            # Keep nonnegative residual diffusion and add the required x drift.
                            qm = remainder/(hm*(hm+hp))
                            qp = remainder/(hp*(hm+hp))
                            qm += (convex - qm*em + qp*ep) / em
                            x_fallback += 1
                    else:
                        qm, qp, fallback = _axial(remainder, bx, hm, hp)
                        x_fallback += int(fallback)
                    q[i-1, j] = q.get((i-1, j), 0.) + qm
                    q[i+1, j] = q.get((i+1, j), 0.) + qp
                if interior_y:
                    qm, qp, fallback = _axial(max(vy-used_y, 0.), by,
                                              y[j]-y[j-1], y[j+1]-y[j])
                    y_fallback += int(fallback)
                    q[i, j-1] = q.get((i, j-1), 0.) + qm
                    q[i, j+1] = q.get((i, j+1), 0.) + qp
                elif zero_vol:
                    h = y[1]-y[0]
                    q[i, 1] = q.get((i, 1), 0.) + by/h
                    zero_vol_nodes += 1
                    zero_vol_added_variance[i, j] = by*h
                elif spatial.reflect_upper_vol and j == ny-1:
                    h = y[-1] - y[-2]
                    q[i, j-1] = 2*coeff.variance_y[i, j]/h**2 + max(-coeff.drift_y[i, j], 0.)/h
                mx = my = mxx = myy = mxy = 0.
                total = 0.
                for (a, b), rate in q.items():
                    if not np.isfinite(rate) or rate < 0:
                        raise ValueError("invalid transition rate")
                    if rate == 0:
                        continue
                    hx, hy = x[a]-x[i], y[b]-y[j]
                    mx += rate*hx
                    my += rate*hy
                    mxx += rate*hx*hx
                    myy += rate*hy*hy
                    mxy += rate*hx*hy
                    rows.append(i*ny+j)
                    columns.append(a*ny+b)
                    entries.append(rate)
                    total += rate
                rows.append(i*ny+j)
                columns.append(i*ny+j)
                entries.append(-total-coeff.discount[i, j])
                errors["drift_x"][i, j] = mx-coeff.drift_x[i, j]
                errors["drift_y"][i, j] = my-coeff.drift_y[i, j]
                errors["variance_x"][i, j] = mxx-2*coeff.variance_x[i, j]
                errors["variance_y"][i, j] = myy-2*coeff.variance_y[i, j]
                errors["covariance_xy"][i, j] = mxy-coeff.covariance_xy[i, j]
                max_covariance_error = max(max_covariance_error, abs(mxy-c))
        backward = sparse.coo_matrix((entries, (rows, columns)), shape=(n, n)).tocsc()
        backward.eliminate_zeros()
        w = grid.quadrature_weights.ravel()
        forward = (sparse.diags(1/w) @ backward.T @ sparse.diags(w)).tocsc()
        constant_error = backward @ np.ones(n) + coeff.discount.ravel()
        exponential = np.exp(grid.x_mesh.ravel())
        martingale = (backward @ exponential) / exponential + coeff.discount.ravel()
        stats = {"nodes": n, "nonzeros": backward.nnz, "negative_off_diagonals": 0,
                 "constant_discount_residual": float(np.max(abs(constant_error))),
                 "max_normalized_forward_generator_error": float(np.max(abs(martingale))),
                 "max_projected_covariance_error": max_covariance_error,
                 "radius_counts": {str(r): int(np.count_nonzero(radius_used == r))
                                   for r in spatial.radii},
                 "diffusion_inflation_nodes": int(inflation.sum()),
                 "boundary_shortened_rays": truncated,
                 "x_drift_fallback_nodes": x_fallback, "y_drift_fallback_nodes": y_fallback,
                 "fit_log_forward": spatial.fit_log_forward,
                 "retain_zero_vol_drift": spatial.retain_zero_vol_drift,
                 "zero_vol_boundary_nodes": zero_vol_nodes,
                 "max_zero_vol_added_variance": float(np.max(zero_vol_added_variance)),
                 "boundary_closure": ("inward drift at degenerate y=0; other normals absorbing"
                                      if zero_vol_nodes else "absorbing normal, evolving tangent"),
                 "max_absolute_moment_errors": {k: float(np.max(abs(v)))
                                                 for k, v in errors.items()}}
        errors["inflation_occupancy"] = inflation.astype(float)
        if spatial.reflect_upper_vol:
            stats["boundary_closure"] = (
                "upper volatility reflected diffusion/inward drift; x normals absorbing; "
                + ("lower volatility inward drift" if zero_vol_nodes else "lower normal absorbing"))
        errors["zero_vol_added_variance"] = zero_vol_added_variance
        return cls(grid, backward, forward, config, stats, moment_errors=errors)

    def step(self, values, direction, step_size, method="be"):
        """Guarantee positivity for BE/exponential, and check the CN positivity restriction."""
        if method == "cn":
            # Sufficient positivity of I+h*A/2. CN is not unconditionally positive.
            bound = step_size * float(np.max(-self.generator(direction).diagonal()))
            if bound > 2.:
                raise ValueError("CN positivity condition violated; use backward Euler")
        return super().step(values, direction, step_size, method)

    def spatial_diagnostics(self, values):
        """Raw density-weighted consistency errors, including the exposed boundary closure."""
        weight = np.abs(values)
        return {f"absolute_{name}": self.grid.integrate(weight*np.abs(error))
                for name, error in self.moment_errors.items()}
