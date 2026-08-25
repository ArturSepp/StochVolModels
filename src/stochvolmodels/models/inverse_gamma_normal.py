"""One-maturity inverse-gamma/normal terminal-law analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral, Real

import numpy as np
import vanilla_option_pricers as bsm
from scipy.optimize import brentq
from scipy.special import ndtr, roots_genlaguerre

from stochvolmodels.data.option_chain import OptionSlice

__all__ = (
    "InverseGammaNormalParams",
    "InverseGammaNormalTerminalModel",
)


@dataclass(frozen=True)
class InverseGammaNormalParams:
    """Risk-neutral parameters of one inverse-gamma/normal terminal law.

    The maturity-specific law is

    ``V ~ IG(alpha, beta)`` and ``X | V ~ N(q * V, c * V)``.

    The shock is measured relative to the prepaid forward. The terminal asset is
    ``S_T = D_T * F_T * max(A_T + X, 0)``, where the model solves ``A_T`` from
    ``E[max(A_T + X, 0)] = 1 / D_T`` for each option slice.
    """

    alpha: float
    beta: float
    c: float
    q: float
    ttm: float


@lru_cache(maxsize=64)
def _gamma_precision_rule(alpha: float, quadrature_order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a read-only quadrature rule for ``U ~ Gamma(alpha, 1)``."""

    nodes, raw_weights = roots_genlaguerre(quadrature_order, alpha - 1.0)
    weight_sum = float(np.sum(raw_weights))
    if (
        not np.all(np.isfinite(nodes))
        or not np.all(nodes > 0.0)
        or not np.all(np.isfinite(raw_weights))
        or np.any(raw_weights < 0.0)
        or not np.isfinite(weight_sum)
        or weight_sum <= 0.0
    ):
        raise ValueError("generalized Gauss-Laguerre quadrature failed for these parameters")
    weights = np.asarray(raw_weights / weight_sum, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _normal_positive_part(mean: np.ndarray, stdev: np.ndarray) -> np.ndarray:
    """Return ``E[(mean + stdev * Z)^+]`` with a stable Gaussian left tail."""

    means = np.asarray(mean, dtype=float)
    stdevs = np.broadcast_to(np.asarray(stdev, dtype=float), means.shape)
    d = means / stdevs
    scaled = np.empty_like(d, dtype=float)
    middle = np.abs(d) <= 8.0
    right = d > 8.0
    left = d < -8.0

    middle_d = d[middle]
    density = np.exp(-0.5 * np.square(middle_d)) / math.sqrt(2.0 * math.pi)
    scaled[middle] = density + middle_d * ndtr(middle_d)
    scaled[right] = d[right]

    z = -d[left]
    inverse_square = 1.0 / np.square(z)
    mills_remainder = inverse_square * (
        1.0
        + inverse_square
        * (-3.0 + inverse_square * (15.0 + inverse_square * (-105.0 + 945.0 * inverse_square)))
    )
    scaled[left] = np.exp(-0.5 * np.square(z)) * mills_remainder / math.sqrt(2.0 * math.pi)
    return np.maximum(stdevs * scaled, 0.0)


class InverseGammaNormalTerminalModel:
    """Validated inverse-gamma normal mean-variance mixture for European options.

    This is a static, one-maturity terminal distribution and Black-smile model. It
    deliberately does not claim path or transform capabilities: for nonzero ``q``
    the marginal law is skewed and heavy-tailed, while at ``q=0`` it reduces to the
    arithmetic Student law without a nontrivial real moment-generating function.

    Parameters are risk-neutral law inputs. Physical parameters and the book's
    ``p``/``eta`` risk-premium mapping remain outside this model.
    """

    def __init__(
        self,
        params: InverseGammaNormalParams,
        *,
        quadrature_order: int = 256,
    ) -> None:
        """Validate and bind one maturity-specific risk-neutral law."""

        if not isinstance(params, InverseGammaNormalParams):
            raise TypeError("params must be an InverseGammaNormalParams instance")

        values = {
            "alpha": params.alpha,
            "beta": params.beta,
            "c": params.c,
            "q": params.q,
            "ttm": params.ttm,
        }
        for name, value in values.items():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite real scalar")
        if values["alpha"] <= 1.0:
            raise ValueError("alpha must be greater than one")
        for name in ("beta", "c", "ttm"):
            if values[name] <= 0.0:
                raise ValueError(f"{name} must be positive")
        if (
            isinstance(quadrature_order, (bool, np.bool_))
            or not isinstance(quadrature_order, Integral)
            or quadrature_order < 2
        ):
            raise ValueError("quadrature_order must be an integer of at least two")

        alpha = float(values["alpha"])
        beta = float(values["beta"])
        c = float(values["c"])
        q = float(values["q"])
        ttm = float(values["ttm"])
        order = int(quadrature_order)
        nodes, weights = _gamma_precision_rule(alpha, order)
        variances = np.asarray(beta / nodes, dtype=float)
        stdevs = np.asarray(np.sqrt(c * variances), dtype=float)
        conditional_means = np.asarray(q * variances, dtype=float)
        mean_mixing_variance = beta / (alpha - 1.0)
        if (
            not np.all(np.isfinite(variances))
            or not np.all(np.isfinite(stdevs))
            or not np.all(np.isfinite(conditional_means))
            or not np.isfinite(mean_mixing_variance)
        ):
            raise ValueError("params produce non-finite quadrature state")
        for array in (variances, stdevs, conditional_means):
            array.setflags(write=False)

        self._alpha = alpha
        self._beta = beta
        self._c = c
        self._q = q
        self._ttm = ttm
        self._quadrature_order = order
        self._weights = weights
        self._variances = variances
        self._stdevs = stdevs
        self._conditional_means = conditional_means
        self._mean_mixing_variance = float(mean_mixing_variance)

    @property
    def params(self) -> InverseGammaNormalParams:
        """Return the immutable risk-neutral parameter payload."""

        return InverseGammaNormalParams(
            alpha=self._alpha,
            beta=self._beta,
            c=self._c,
            q=self._q,
            ttm=self._ttm,
        )

    @property
    def ttm(self) -> float:
        """Return the terminal law's exact maturity in years."""

        return self._ttm

    @property
    def quadrature_order(self) -> int:
        """Return the generalized Gauss-Laguerre node count."""

        return self._quadrature_order

    @property
    def mean_mixing_variance(self) -> float:
        """Return ``E[V] = beta / (alpha - 1)`` under the bound pricing law."""

        return self._mean_mixing_variance

    def _expect_positive(
        self,
        intercept: float | np.ndarray,
        *,
        shock_sign: float = 1.0,
    ) -> np.ndarray:
        intercepts = np.asarray(intercept, dtype=float)
        means = intercepts[..., np.newaxis] + shock_sign * self._conditional_means
        conditional = _normal_positive_part(means, self._stdevs)
        return np.sum(conditional * self._weights, axis=-1)

    @staticmethod
    def _validate_discfactor(discfactor: float) -> float:
        if (
            isinstance(discfactor, (bool, np.bool_))
            or not isinstance(discfactor, Real)
            or not np.isfinite(discfactor)
            or discfactor <= 0.0
        ):
            raise ValueError("discfactor must be a finite positive real scalar")
        return float(discfactor)

    @lru_cache(maxsize=32)
    def _solve_martingale_shift(self, discfactor: float) -> float:
        target = 1.0 / discfactor

        def objective(shift: float) -> float:
            return float(self._expect_positive(shift)) - target

        center = target - self._q * self._mean_mixing_variance
        radius = max(
            1.0,
            abs(center),
            math.sqrt(self._c * self._mean_mixing_variance),
        )
        for _ in range(128):
            lower = center - radius
            upper = center + radius
            lower_value = objective(lower)
            upper_value = objective(upper)
            if lower_value <= 0.0 <= upper_value:
                break
            radius *= 2.0
            if not np.isfinite(radius):
                raise ValueError("could not bracket the martingale shift")
        else:
            raise ValueError("could not bracket the martingale shift")

        return float(
            brentq(
                objective,
                lower,
                upper,
                xtol=1.0e-13,
                rtol=1.0e-13,
                maxiter=200,
            )
        )

    def martingale_shift(self, *, discfactor: float) -> float:
        """Return the unique prepaid-forward shift for ``discfactor``."""

        validated = self._validate_discfactor(discfactor)
        return self._solve_martingale_shift(validated)

    def default_probability(self, *, discfactor: float) -> float:
        """Return the terminal zero-price mass for ``discfactor``."""

        shift = self.martingale_shift(discfactor=discfactor)
        standardized_boundary = (-shift - self._conditional_means) / self._stdevs
        return float(np.dot(self._weights, ndtr(standardized_boundary)))

    def _validated_slice(
        self,
        option_slice: OptionSlice,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        if not isinstance(option_slice, OptionSlice):
            raise TypeError("option_slice must be an OptionSlice instance")
        slice_ttm = option_slice.ttm
        if (
            isinstance(slice_ttm, (bool, np.bool_))
            or not isinstance(slice_ttm, Real)
            or not np.isfinite(slice_ttm)
            or slice_ttm <= 0.0
        ):
            raise ValueError("option_slice.ttm must be a finite positive real scalar")
        if float(slice_ttm) != self.ttm:
            raise ValueError("option_slice.ttm must exactly match the bound params.ttm")

        forward = option_slice.forward
        if (
            isinstance(forward, (bool, np.bool_))
            or not isinstance(forward, Real)
            or not np.isfinite(forward)
            or forward <= 0.0
        ):
            raise ValueError("option_slice.forward must be a finite positive real scalar")
        discfactor = self._validate_discfactor(option_slice.discfactor)

        raw_strikes = np.asarray(option_slice.strikes)
        if raw_strikes.dtype.kind == "b":
            raise ValueError("option_slice.strikes must contain finite positive real values")
        strikes = np.asarray(option_slice.strikes, dtype=float)
        optiontypes = np.asarray(option_slice.optiontypes).astype(str)
        if (
            strikes.ndim != 1
            or strikes.size == 0
            or optiontypes.ndim != 1
            or optiontypes.size != strikes.size
            or not np.all(np.isfinite(strikes))
            or np.any(strikes <= 0.0)
        ):
            raise ValueError("option_slice strikes and optiontypes must be aligned and valid")
        unsupported = set(optiontypes) - {"C", "P"}
        if unsupported:
            raise NotImplementedError(
                "InverseGammaNormalTerminalModel supports C/P only; "
                "inverse IC/IP settlement is not implemented"
            )
        return float(forward), discfactor, strikes, optiontypes

    def price_european(self, option_slice: OptionSlice) -> np.ndarray:
        """Return discounted standard-call/put prices for one exact-maturity slice."""

        forward, discfactor, strikes, optiontypes = self._validated_slice(option_slice)
        prepaid_forward = discfactor * forward
        normalized_strikes = strikes / prepaid_forward
        shift = self.martingale_shift(discfactor=discfactor)

        normalized_prices = np.empty_like(normalized_strikes, dtype=float)
        call_mask = optiontypes == "C"
        if np.any(call_mask):
            normalized_prices[call_mask] = self._expect_positive(
                shift - normalized_strikes[call_mask]
            )
        put_mask = ~call_mask
        if np.any(put_mask):
            raw_puts = self._expect_positive(
                normalized_strikes[put_mask] - shift,
                shock_sign=-1.0,
            )
            floor_adjustment = float(self._expect_positive(-shift, shock_sign=-1.0))
            normalized_prices[put_mask] = raw_puts - floor_adjustment
        return np.asarray(discfactor * prepaid_forward * normalized_prices, dtype=float)

    def implied_vols(self, option_slice: OptionSlice) -> np.ndarray:
        """Return Black implied volatilities shaped like ``option_slice.strikes``."""

        forward, discfactor, strikes, optiontypes = self._validated_slice(option_slice)
        prices = self.price_european(option_slice)
        ivols = bsm.infer_bsm_ivols_from_slice_prices(
            ttm=self.ttm,
            forward=forward,
            discfactor=discfactor,
            strikes=strikes,
            optiontypes=optiontypes,
            model_prices=prices,
        )
        return np.asarray(ivols, dtype=float)
