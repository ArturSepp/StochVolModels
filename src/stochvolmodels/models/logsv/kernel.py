"""LogSV kernel calculations delegated by ``LogSvParams``.

These model-specific computations preserve the parameter convenience API while
keeping numerical implementations separate from parameter definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from stochvolmodels.models.rough_logsv.kernel_fit import KernelRule, moment_fitted_kernel
from stochvolmodels.models.rough_logsv.rough_kernel import european_rule

if TYPE_CHECKING:
    from stochvolmodels.models.logsv.params import LogSvParams


def approximate_kernel(
    params: LogSvParams,
    T: float,
    kernel_rule: KernelRule = KernelRule.EUROPEAN,
    w_bar_max: Optional[float] = None,
) -> None:
    """
    set the Markovian approximation nodes and weights of the rough kernel.

    Outside the scope of the article, which fixes H = 1/2. The rough model replaces the
    constant kernel by K(t) = t^(H-1/2) / Gamma(H+1/2) and approximates it by
    K^N(t) = sum_i weights_i exp(-nodes_i t) over [0, T]. The number of nodes is 1 for H in
    (0.49, 0.5], 2 for H in (0.4, 0.49] and 3 below; H in (0.49, 0.5] uses the single
    node 1e-3 with weight 1, so K^N(t) = exp(-1e-3 t), which reproduces the standard
    (non-rough) dynamics of Eq. (3.12) up to a mean reversion of rate 1e-3.

    The default rule is the European quadrature rule of Bayer and Breneis (2023). The rule
    ``KernelRule.MOMENT_FIT`` fits the nodes and weights to the mean, variance-swap
    volatility and leverage of the model with kappa2 = 0 (``rough_logsv.kernel_fit``). It
    reduces the kernel error for H <= 0.2 but raises w_bar = sum_i weights_i, and the
    empirical split-step resolution guideline is w_bar vartheta sqrt(h) <= 0.4; pass ``w_bar_max``
    from ``kernel_fit.max_weight_sum`` to fit for a given step. The first moment fit for a
    given (H, T, w_bar_max) takes about a minute and is memoised.

    Parameters
    ----------
    T : float
        Horizon in years over which the kernel is approximated.
    kernel_rule : KernelRule, default KernelRule.EUROPEAN
        Approximation rule for H <= 0.49.
    w_bar_max : float, optional
        Required final cap on w_bar for ``KernelRule.MOMENT_FIT``. The penalized optimizer
        can fail to achieve it, in which case it raises. Ignored by the European rule.
    """
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("kernel horizon T must be finite and positive")
    if kernel_rule not in (KernelRule.EUROPEAN, KernelRule.MOMENT_FIT):
        raise ValueError(f"unknown kernel_rule {kernel_rule}")
    if kernel_rule == KernelRule.MOMENT_FIT and w_bar_max is not None:
        if not np.isfinite(w_bar_max) or w_bar_max <= 0.0:
            raise ValueError("w_bar_max must be finite and positive")
        if params.H > 0.49 and w_bar_max < 1.0:
            raise ValueError("the near-Markov unit-weight kernel exceeds w_bar_max")
    if 0.49 < params.H <= 0.5:
        params.weights = np.array([1.0])
        params.nodes = np.array([1e-3])
        return
    elif 0.4 < params.H <= 0.49:
        N = 2
    else:
        N = 3
    if kernel_rule == KernelRule.MOMENT_FIT:
        params.nodes, params.weights = moment_fitted_kernel(
            hurst=params.H, n_nodes=N, horizon=T, w_bar_max=w_bar_max
        )
    elif kernel_rule == KernelRule.EUROPEAN:
        params.nodes, params.weights = european_rule(params.H, N, T)
    else:
        raise ValueError(f"unknown kernel_rule {kernel_rule}")
