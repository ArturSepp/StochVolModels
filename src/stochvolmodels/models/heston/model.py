"""Heston model definitions.

Parameter-bound Heston transforms and paths for the provisional model interface.

Paths describe a zero-drift asset under the money-market-account measure, with time
in years, instantaneous annualized variance, annualized volatility and cumulative
integrated variance recorded separately. Historical Euler uses a 1e-4 variance
floor. Optional QE-M uses the nonnegative quadratic-exponential transition.
"""

from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral, Real
import numpy as np
from numpy.typing import NDArray
from stochvolmodels.data.model_paths import ModelPaths

from .params import HestonParams


def _real(value: object, name: str, *, nonnegative: bool = True) -> float:
    """Validate a finite real scalar, rejecting booleans and optionally negatives."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{name} must be finite" + (" and non-negative" if nonnegative else ""))
    return result


def _integer(value: object, name: str, *, minimum: int = 1) -> int:
    """Validate an integer count or seed without silently truncating floats."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _readonly(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Freeze a producer-owned simulation array."""
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True, init=False)
class HestonModel:
    """Validated Heston dynamics exposing path and logarithmic-transform capabilities.

    Parameters are copied into an immutable scalar snapshot. The Feller condition is
    not imposed: it is a boundary property, not a prerequisite for the Heston process.
    ``HestonParams`` is defined here; historical pricer imports remain aliases.
    Transform and simulation kernels are imported lazily from the model layer.
    """

    _v0: float
    _theta: float
    _kappa: float
    _rho: float
    _volvol: float

    def __init__(self, params: HestonParams) -> None:
        """Validate and snapshot variance parameters and return-variance correlation."""
        from stochvolmodels.models.heston.params import HestonParams

        if not isinstance(params, HestonParams):
            raise ValueError("params must be a HestonParams instance")
        for name in ("v0", "theta", "kappa", "volvol"):
            object.__setattr__(self, f"_{name}", _real(getattr(params, name), name))
        rho = _real(params.rho, "rho", nonnegative=False)
        if abs(rho) > 1.0:
            raise ValueError("rho must be in [-1, 1]")
        object.__setattr__(self, "_rho", rho)

    @property
    def params(self) -> HestonParams:
        """Return a detached parameter object without exposing the stored snapshot."""
        from stochvolmodels.models.heston.params import HestonParams

        return HestonParams(self._v0, self._theta, self._kappa, self._rho, self._volvol)

    def log_mgf_grid(
        self,
        *,
        ttm: float,
        phi_grid: np.ndarray,
        psi_grid: np.ndarray | None = None,
    ) -> NDArray[np.complex128]:
        """Return ``log E[exp(-phi * X_t - psi * I_t)]`` under ``Q_MMA``.

        ``X_t`` is the zero-drift asset's log return and ``I_t`` its integrated
        variance. Time is in years; transform grids must be finite one-dimensional
        arrays of identical shape. Omitting ``psi_grid`` selects the return transform.
        Zero maturity and deterministic variance use their exact limiting formulas.
        The legacy closed form handles ordinary grids. Near cancellation, repeated
        Riccati roots or very small volatility of variance, an adaptive Riccati solve
        evaluates the same dynamics without subtracting nearly equal coefficients.
        """
        intercept, loading = self._affine_coefficients(ttm=ttm, phi_grid=phi_grid, psi_grid=psi_grid)
        result = intercept + self._v0 * loading
        if not np.isfinite(result).all():
            raise FloatingPointError("Heston transform is not finite on the requested grid")
        return result

    def _affine_coefficients(
        self, *, ttm: float, phi_grid: np.ndarray, psi_grid: np.ndarray | None = None,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Return the exact affine coefficients A, B in log M = A + B*v0."""
        maturity = _real(ttm, "ttm")
        grids = []
        for name, grid in (("phi_grid", phi_grid), ("psi_grid", psi_grid)):
            if grid is None and name == "psi_grid":
                grids.append(np.zeros_like(grids[0]))
                continue
            if not isinstance(grid, np.ndarray) or grid.dtype.kind not in "iufc":
                raise ValueError(f"{name} must be a numeric NumPy array")
            if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all():
                raise ValueError(f"{name} must be a non-empty finite one-dimensional grid")
            grids.append(np.asarray(grid, dtype=np.complex128))
        phi, psi = grids
        if phi.shape != psi.shape:
            raise ValueError("phi_grid and psi_grid must have the same shape")
        intercept, loading = np.zeros_like(phi), np.zeros_like(phi)
        if maturity == 0.0:
            return intercept, loading
        b0 = 0.5 * phi * (phi + 1.0) - psi
        if self._volvol == 0.0:
            kt = self._kappa * maturity
            duration = maturity if kt == 0.0 else -np.expm1(-kt) / self._kappa
            # Resolve the small long-run-variance contribution even at v0 = 0.
            drift_duration = (
                maturity * kt * (0.5 - kt / 6 + kt**2 / 24)
                if kt < 1e-5 else maturity - duration
            )
            return b0 * (self._theta * drift_duration), b0 * duration

        from stochvolmodels.models.heston.kernels import _heston_affine_coefficients

        # Every zero constant term gives the exact zero Riccati solution, including
        # joint return/variance grids beyond the usual phi=0 and phi=-1 anchors.
        active = b0 != 0.0
        if not np.any(active):
            return intercept, loading

        volvol2 = self._volvol * self._volvol
        b1 = self._kappa + self._rho * self._volvol * phi
        # These dimensionless conditions detect cancellation in zeta-b1, exp(-zeta*t),
        # and the closed-form denominator. They choose an integration algorithm;
        # they do not replace a small parameter by zero or approximate the model.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            diffusion_scale = 2.0 * volvol2 * np.abs(b0)
            drift_scale = np.abs(b1) ** 2
            discriminant = b1 * b1 - 2.0 * volvol2 * b0
            zeta = np.sqrt(discriminant)
            plus_term = (zeta - b1) * np.exp(-zeta * maturity)
            minus_term = b1 + zeta
            denominator = plus_term + minus_term
            denominator_scale = np.maximum(
                np.abs(plus_term) + np.abs(minus_term), np.finfo(float).tiny
            )
            fallback = active & (
                (diffusion_scale <= 1.0e-6 * drift_scale)
                | (np.abs(zeta * maturity) <= 1.0e-4)
                | (np.abs(discriminant) <= 1.0e-8 * (drift_scale + diffusion_scale))
                | (np.abs(denominator) <= 1.0e-8 * denominator_scale)
                | ~np.isfinite(denominator)
            )
        direct = active & ~fallback
        if np.any(direct):
            try:
                a, b = _heston_affine_coefficients(
                    theta=self._theta,
                    kappa=self._kappa,
                    rho=self._rho,
                    volvol=self._volvol,
                    ttm=maturity,
                    phi_grid=phi[direct],
                    psi_grid=psi[direct],
                )
            except ArithmeticError:
                fallback[direct] = True
            else:
                indices = np.flatnonzero(direct)
                finite = np.isfinite(a) & np.isfinite(b)
                intercept[indices[finite]], loading[indices[finite]] = a[finite], b[finite]
                fallback[indices[~finite]] = True

        if np.any(fallback):
            from scipy.integrate import solve_ivp

            for index in np.flatnonzero(fallback):
                constant, linear = b0[index], b1[index]

                def rhs(time: float, state: NDArray[np.complex128]) -> NDArray[np.complex128]:
                    """Evaluate affine Riccati equations without closed-form divisions."""
                    b = state[1]
                    return np.array(
                        [
                            self._kappa * self._theta * b,
                            0.5 * volvol2 * b * b - linear * b + constant,
                        ],
                        dtype=np.complex128,
                    )

                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    solution = solve_ivp(
                        rhs,
                        (0.0, maturity),
                        np.zeros(2, dtype=np.complex128),
                        method="DOP853",
                        rtol=1.0e-11,
                        atol=1.0e-13,
                    )
                if not solution.success or not np.isfinite(solution.y[:, -1]).all():
                    raise FloatingPointError(
                        "Heston Riccati transform failed on the requested grid"
                    )
                intercept[index], loading[index] = solution.y[:, -1]
        return intercept, loading

    def simulate_paths(
        self,
        *,
        measure: str,
        observation_times: np.ndarray,
        spot0: float,
        n_paths: int,
        steps_per_year: int,
        seed: int,
        mc_scheme: str = "euler",
    ) -> ModelPaths:
        """Simulate zero-drift Heston paths using local, reproducible random draws.

        Parameters
        ----------
        measure
            ``Q_MMA``; inverse and physical measures are unsupported.
        observation_times
            Finite strictly increasing times in years, beginning at zero. QE-M also
            accepts the initial observation alone for an expiry at time zero.
        spot0
            Positive initial level of the zero-drift asset (no forward carry).
        n_paths, steps_per_year
            Positive path count and step density. Each observation interval uses
            the legacy rule ``int(interval * steps_per_year) + 1``.
        seed
            Integer in ``[0, 2**32 - 1]`` for a local NumPy ``RandomState``.
        mc_scheme
            Historical Euler or ``"qe_m"`` for Andersen's martingale-corrected QE scheme.

        Returns
        -------
        ModelPaths
            Read-only arrays with shape ``(n_paths, n_times, 1)`` for assets and
            ``(n_paths, n_times)`` for states. Pricing, discounting, and optional
            forward recentering belong to the valuation layer. The observation
            partition affects both the integration grid and the draw ordering.
        """
        if mc_scheme not in ("euler", "qe_m"):
            raise ValueError("mc_scheme must be 'euler' or 'qe_m'")
        if measure != "Q_MMA":
            raise ValueError("HestonModel supports only measure='Q_MMA'")
        if (
            not isinstance(observation_times, np.ndarray)
            or observation_times.dtype.kind not in "iuf"
        ):
            raise ValueError("observation_times must be a real numeric NumPy array")
        times = np.array(observation_times, dtype=np.float64, copy=True)
        if times.ndim != 1 or times.size < (1 if mc_scheme == "qe_m" else 2) or not np.isfinite(times).all():
            raise ValueError("observation_times must be a finite one-dimensional grid")
        if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("observation_times must begin at zero and strictly increase")
        spot = _real(spot0, "spot0")
        if spot == 0.0:
            raise ValueError("spot0 must be strictly positive")
        count = _integer(n_paths, "n_paths")
        density = _integer(steps_per_year, "steps_per_year")
        seed_value = _integer(seed, "seed", minimum=0)
        if seed_value >= 2**32:
            raise ValueError("seed must be in [0, 2**32 - 1]")

        from stochvolmodels.models.heston.kernels import _qe_m_step, _simulate_heston_with_increments
        from stochvolmodels.utils.funcs import set_time_grid

        rng = np.random.RandomState(seed_value)
        shape = (count, times.size)
        returns, variances, integrated = (np.empty(shape) for _ in range(3))
        returns[:, 0], variances[:, 0], integrated[:, 0] = 0.0, self._v0, 0.0
        interval_steps: list[int] = []
        interval_dts: list[float] = []
        for index, interval in enumerate(np.diff(times), start=1):
            steps, dt, _ = set_time_grid(float(interval), density)
            x, variance, qvar = (state[:, index - 1].copy()
                                 for state in (returns, variances, integrated))
            if mc_scheme == "qe_m":
                for _ in range(steps):
                    z_var, z_spot = rng.standard_normal((2, count))
                    _qe_m_step(x, variance, qvar, self._theta, self._kappa, self._rho,
                               self._volvol, dt, z_var, z_spot)
            else:
                w0 = np.sqrt(dt) * rng.normal(size=(steps, count))
                w1 = np.sqrt(dt) * rng.normal(size=(steps, count))
                x, variance, qvar = _simulate_heston_with_increments(
                    x, variance, qvar, self._theta, self._kappa, self._rho, self._volvol, dt, w0, w1,
                )
            returns[:, index], variances[:, index], integrated[:, index] = x, variance, qvar
            interval_steps.append(int(steps))
            interval_dts.append(float(dt))

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            assets = (spot * np.exp(returns))[:, :, np.newaxis].copy()
            sigma = np.sqrt(variances)
        failed = (
            np.any(~np.isfinite(assets) | (assets <= 0.0), axis=(1, 2))
            | np.any(~np.isfinite(returns), axis=1)
            | np.any(~np.isfinite(variances) | (variances < 0.0), axis=1)
            | np.any(~np.isfinite(integrated) | (integrated < 0.0), axis=1)
        )
        if mc_scheme == "qe_m" and np.any(failed):
            raise FloatingPointError("nonfinite or nonpositive Heston QE-M path value")
        qe = mc_scheme == "qe_m"
        return ModelPaths(
            observation_times=_readonly(times),
            assets=_readonly(assets),
            asset_ids=("zero_drift_price",),
            sampling_measure="Q_MMA",
            target_measure="Q_MMA",
            numeraire="money_market_account",
            scheme=("heston_qe_m_trapezoidal_qvar_v1" if qe
                    else "heston_euler_variance_floor_1e-4_left_qvar_v1"),
            states={
                "log_zero_drift_return": _readonly(returns),
                "variance": _readonly(variances),
                "sigma": _readonly(sigma),
                "quadratic_variance": _readonly(integrated),
            },
            state_units={
                "log_zero_drift_return": "log return",
                "variance": "annualized variance",
                "sigma": "annualized volatility",
                "quadratic_variance": "integrated variance",
            },
            provenance={
                "steps_per_year": density,
                "interval_step_counts": tuple(interval_steps),
                "interval_dts": tuple(interval_dts),
                "seed": seed_value,
                "generator": "numpy.random.RandomState",
                "bit_generator": "MT19937",
                "draw_ordering": ("per_step_variance_then_orthogonal_spot" if qe
                                  else "per_interval_W0_matrix_then_W1_matrix_row_major"),
                "normal_convention": "standard_normals" if qe else "scaled_Brownian_increments",
                "observation_partition_dependent": True,
                "variance_floor": 0.0 if qe else 1e-4,
                "numpy_version": np.__version__,
            },
            diagnostics={
                "failed_path_count": int(np.count_nonzero(failed)),
                "qvar_decrease_count": int(np.count_nonzero(np.diff(integrated, axis=1) < 0.0)),
            },
        )
