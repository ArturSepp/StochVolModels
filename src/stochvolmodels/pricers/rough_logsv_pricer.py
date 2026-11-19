"""Standalone experimental rough-logSV pricer for the validated paper workflow.

The H=1/2 limit uses the full nonlinear Fourier/backward PDE. Rough prices use
the complete factor history, adapted price integrals, a validated discrete
classical control and independent multilevel scrambles. No transform formula
for the general rough model is claimed. Wider calibration remains gated.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from stochvolmodels.pricers.model_pricer import ModelPricer
from stochvolmodels.models.rough_logsv import _classical_pde
from stochvolmodels.models.rough_logsv.multilevel import (
    DiscretePdeControl,
    RoughMCConfig,
    numerical_checks,
    sample_multilevel,
)
from stochvolmodels.models.rough_logsv.params import RoughKernel, RoughLogSvParams
from stochvolmodels.utils.config import VariableType


@dataclass(frozen=True)
class ClassicalPdeConfig:
    """Explicit space/time/domain/contour settings for the exact H=1/2 model."""

    subdivisions: int = 160
    halfwidth: float = 6.0
    steps_per_day: int = 8
    degree: int = 384
    cutoff: float = 256.0
    formulation: str = "direct"

    def __post_init__(self):
        """Reject invalid PDE settings before invoking compiled numerical kernels."""
        for name in ("subdivisions", "steps_per_day", "degree"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not np.isfinite([self.halfwidth, self.cutoff]).all()
            or min(self.halfwidth, self.cutoff) <= 0
            or self.formulation not in ("direct", "gauge")
        ):
            raise ValueError(
                "Positive finite domain/contour and direct or gauge formulation required"
            )
        if round(2 * self.halfwidth * self.subdivisions) % 2:
            raise ValueError("PDE grid must have an even number of intervals")


@dataclass
class RoughPricingResult:
    """Prices, sampling errors and numerical diagnostics in explicit units.

    prices and standard_errors are discounted currency premiums, grouped by
    expiry. normalized_summary is in forward-normalized undiscounted units.
    PDE results have no Monte Carlo standard error. Numerical checks never
    certify an optimizer's parameter domain.
    """

    prices: list
    standard_errors: list | None
    normalized_summary: dict
    diagnostics: dict
    calibration_ready: bool = False


def chain_coordinates(chain):
    """Validate European C/P quotes and convert ACT/365 maturities to exact days.

    The integrated paper workflow currently supports integer-day maturities.
    Fractional days are rejected, never silently rounded. Quote ordering is retained.
    """
    times = np.asarray(chain.ttms, float)
    days_float = times * 365
    if (
        times.ndim != 1
        or len(times) == 0
        or not np.isfinite(times).all()
        or times[0] <= 0
        or np.any(np.diff(times) <= 0)
        or not np.allclose(days_float, np.rint(days_float), rtol=0, atol=1e-9)
    ):
        raise ValueError("Increasing positive integer-day ACT/365 maturities required")
    n = len(times)
    if any(
        len(x) != n
        for x in (chain.forwards, chain.discfactors, chain.strikes_ttms, chain.optiontypes_ttms)
    ):
        raise ValueError("Option-chain expiry arrays must have the same length")
    strikes, calls, expiry, scales, sizes = [], [], [], [], []
    for j, (forward, discount, k, kind) in enumerate(
        zip(chain.forwards, chain.discfactors, chain.strikes_ttms, chain.optiontypes_ttms)
    ):
        k, kind = np.asarray(k, float), np.asarray(kind)
        if (
            not np.isfinite([forward, discount]).all()
            or min(forward, discount) <= 0
            or k.ndim != 1
            or len(k) == 0
            or kind.shape != k.shape
            or not np.isfinite(k).all()
            or np.any(k <= 0)
            or not np.isin(kind, ["C", "P"]).all()
        ):
            raise ValueError("Positive forwards/strikes/discounts and European C/P quotes required")
        strikes.extend(k / forward)
        calls.extend(kind == "C")
        expiry.extend([j] * len(k))
        scales.extend([forward * discount] * len(k))
        sizes.append(len(k))
    return (
        np.rint(days_float).astype(np.int64),
        np.array(expiry, int),
        np.array(strikes),
        np.array(calls, bool),
        np.array(scales),
        sizes,
    )


class RoughLogSVPricer(ModelPricer):
    """Independent experimental interface; no rough-mode flag on LogSVPricer.

    For H<1/2, supply the finite kernel, a comparison kernel, a fixed MC
    allocation and an independently validated discrete control mean.
    The selected BTC/ETH benchmark runner supplies the archived inputs.
    This class never imports paper scripts, agent bundles or private data.
    """

    def __init__(
        self,
        kernel: RoughKernel | None = None,
        comparison_kernel: RoughKernel | None = None,
        mc_config: RoughMCConfig | None = None,
        control: DiscretePdeControl | None = None,
        pde_config: ClassicalPdeConfig | None = None,
    ):
        super().__init__()
        self.kernel = kernel
        self.comparison_kernel = comparison_kernel
        self.mc_config = mc_config
        self.control = control
        self.pde_config = pde_config or ClassicalPdeConfig()

    @staticmethod
    def _validate(params, variable_type, is_spot_measure, kwargs):
        if kwargs:
            raise TypeError(f"Unsupported rough-logSV options: {', '.join(sorted(kwargs))}")
        if not isinstance(params, RoughLogSvParams):
            raise TypeError("Use RoughLogSvParams, not LogSvParams")
        params.__post_init__()
        if variable_type != VariableType.LOG_RETURN or not is_spot_measure:
            raise NotImplementedError(
                "Rough-logSV supports vanilla money-market-measure pricing only"
            )
        if params.martingale_margin < 0:
            raise ValueError(
                "Bounded option estimator requires kappa2 >= beta; use raw diagnostics separately"
            )

    def price_chain_result(
        self,
        option_chain,
        params: RoughLogSvParams,
        variable_type=VariableType.LOG_RETURN,
        is_spot_measure=True,
        full_spreads=None,
        **kwargs,
    ):
        """Price a chain and retain diagnostics; spreads use currency-premium units.

        H=1/2 uses the nonlinear PDE. All other H use the explicitly configured
        multilevel estimator. Numerical comparisons qualify only the evaluated
        coordinates and never open wider calibration.
        """
        self._validate(params, variable_type, is_spot_measure, kwargs)
        days, expiry, strikes, calls, scales, sizes = chain_coordinates(option_chain)
        spread = None
        if full_spreads is not None:
            spread = np.asarray(full_spreads, float)
            if spread.shape != scales.shape or not np.isfinite(spread).all() or np.any(spread <= 0):
                raise ValueError(
                    "Positive full spreads must be flat and follow chain quote ordering"
                )
        diagnostics = dict(
            H=params.H,
            martingale_margin=params.martingale_margin,
            backbone=1.0,
            forward_normalization=False,
            parameter_domain_qualified=False,
        )
        if params.vartheta == 0:
            if params.theta != params.sigma0 and (params.kappa1 or params.kappa2):
                raise NotImplementedError(
                    "Zero-noise pricing currently requires constant volatility"
                )
            root = params.sigma0 * np.sqrt(np.asarray(option_chain.ttms)[expiry])
            d = -np.log(strikes) / root + root / 2
            price = ndtr(d) - strikes * ndtr(d - root) - np.where(calls, 0, 1 - strikes)
            summary = dict(
                price=price,
                standard_error=np.zeros_like(price),
                price_halfwidth=np.zeros_like(price),
                raw_forward=np.ones(len(days)),
            )
            diagnostics["method"] = "constant_volatility"
        elif params.H == 0.5:
            if self.kernel is not None and self.kernel != RoughKernel.classical():
                raise ValueError("H=0.5 uses the exact constant kernel")
            p = self.pde_config
            u, w = _classical_pde.contour(p.degree, p.cutoff)
            transforms = _classical_pde.transforms(
                _classical_pde.from_row(params),
                0.5 - 1j * u,
                days,
                subdivisions=p.subdivisions,
                halfwidth=p.halfwidth,
                per_day=p.steps_per_day,
                formulation=p.formulation,
            )
            price = np.empty(len(strikes))
            for j in range(len(days)):
                mask = expiry == j
                price[mask] = _classical_pde.calls(u, w, transforms[j], strikes[mask]) - np.where(
                    calls[mask], 0, 1 - strikes[mask]
                )
            if not np.isfinite(price).all():
                raise FloatingPointError("Nonfinite classical PDE prices")
            summary = dict(price=price)
            diagnostics.update(method="nonlinear_classical_pde", refinement_required=True)
        else:
            if any(
                x is None
                for x in (self.kernel, self.comparison_kernel, self.mc_config, self.control)
            ):
                raise ValueError(
                    "Rough pricing requires explicit kernels, allocation and validated discrete control"
                )
            run = sample_multilevel(
                params,
                self.kernel,
                self.comparison_kernel,
                self.mc_config,
                self.control,
                days,
                expiry,
                strikes,
                calls,
            )
            summary = run.pop("summary")
            diagnostics.update(run, method="history_multilevel_conditional")
            if spread is not None:
                diagnostics["checks"] = numerical_checks(summary, spread / scales)
        boundaries = np.cumsum(sizes)[:-1]
        prices = list(np.split(summary["price"] * scales, boundaries))
        errors = (
            list(np.split(summary["standard_error"] * scales, boundaries))
            if "standard_error" in summary
            else None
        )
        return RoughPricingResult(prices, errors, summary, diagnostics)

    def price_chain(self, option_chain, params: RoughLogSvParams, **kwargs):
        """Return premiums through the shared ModelPricer chain interface."""
        return self.price_chain_result(option_chain, params, **kwargs).prices

    def model_mc_price_chain(self, option_chain, params: RoughLogSvParams, **kwargs):
        """Return premiums and MC standard errors for the rough conditional estimator."""
        if isinstance(params, RoughLogSvParams) and params.H == 0.5 and params.vartheta:
            raise NotImplementedError("H=0.5 uses the deterministic PDE; call price_chain_result")
        result = self.price_chain_result(option_chain, params, **kwargs)
        if result.standard_errors is None:
            raise NotImplementedError("H=0.5 uses the deterministic PDE; call price_chain_result")
        return result.prices, result.standard_errors

    def calibrate_model_params_to_chain(self, option_chain, **kwargs):
        """Keep optimization gated until an applicable parameter-domain protocol exists."""
        raise NotImplementedError(
            "Wider rough-logSV calibration remains gated. Saved BTC/ETH fits qualify their "
            "evaluated parameters, not an optimizer domain; use the selected-case replay runner."
        )
