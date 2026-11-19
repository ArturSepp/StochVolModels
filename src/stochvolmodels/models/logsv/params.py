"""
Parameters of the log-normal beta SV model with quadratic drift.

Container for the six parameters of the volatility process in Eq. (3.12) of
Sepp and Rakhmonov (2024),

    dsigma_t = (kappa1 + kappa2 sigma_t)(theta - sigma_t) dt
               + beta sigma_t dW^(0)_t + epsilon sigma_t dW^(1)_t,

together with the derived quantities the affine expansion and the moment ODEs
consume: the total volatility of volatility vartheta^2 = beta^2 + epsilon^2 in
Eq. (3.13), the effective mean-reversion kappa = kappa1 + kappa2 theta in
Eq. (3.32), and the truncated moment generator Lambda^(1,k*) in Eq. (3.48).

Reference
---------
A. Sepp and P. Rakhmonov (2024), Log-normal Stochastic Volatility Model with
Quadratic Drift, International Journal of Theoretical and Applied Finance 26(8),
2450003. Equation numbers throughout this module refer to that article.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from stochvolmodels.models.base import ModelParams
from stochvolmodels.models.rough_logsv.kernel_fit import KernelRule
from stochvolmodels.utils.config import VariableType
from stochvolmodels.utils.funcs import find_nearest


@dataclass
class LogSvParams(ModelParams):
    """
    Parameters of the log-normal beta SV model with quadratic drift.

    The volatility process of Eq. (3.12) under the MMA measure Q is

        dsigma_t = (kappa1 + kappa2 sigma_t)(theta - sigma_t) dt
                   + beta sigma_t dW^(0)_t + epsilon sigma_t dW^(1)_t.

    Attributes
    ----------
    sigma0 : float, default 0.2
        Initial volatility sigma_0.
    theta : float, default 0.2
        Mean volatility theta > 0.
    kappa1 : float, default 1.0
        Linear mean-reversion rate kappa1 >= 0.
    kappa2 : Optional[float], default 2.5
        Quadratic mean-reversion rate kappa2 >= 0. ``None`` maps to
        ``kappa1 / theta``, the pure quadratic drift case. Theorem 3.7 requires
        ``kappa2 >= beta`` for Z_t to be a Q-martingale and ``kappa2 >= 2 beta``
        for R_t to be a martingale under the inverse measure; neither bound is
        enforced here.
    beta : float, default -1.0
        Volatility beta, the loading of the volatility on the price Brownian
        motion. Sign follows the return-volatility correlation.
    volvol : float, default 1.0
        Volatility of residual volatility epsilon > 0.
    vol_backbone : pd.Series, default None
        Optional deterministic loading eta on the price diffusion, indexed by
        time to maturity; theta and the volatility-state drift remain unchanged.
        The rough simulator currently supports eta = 1 only.
    H : float, default 0.5
        Hurst exponent. 0.5 is the article's dynamics; values below 0.5 select the
        rough extension noted in Sec. 7.
    weights : np.ndarray, default None
        Rough-kernel quadrature weights, set by :meth:`approximate_kernel`.
    nodes : np.ndarray, default None
        Rough-kernel quadrature nodes, set by :meth:`approximate_kernel`.
    """

    sigma0: float = 0.2
    theta: float = 0.2
    kappa1: float = 1.0
    kappa2: Optional[float] = 2.5  # Optional is mapped to self.kappa1 / self.theta
    beta: float = -1.0
    volvol: float = 1.0
    vol_backbone: pd.Series = None
    H: float = 0.5
    weights: np.ndarray = None
    nodes: np.ndarray = None

    def __post_init__(self):
        """
        map the optional quadratic mean-reversion rate and validate the Hurst exponent.

        A ``kappa2`` of ``None`` is mapped to ``kappa1 / theta``, the pure quadratic
        drift case ``kappa2 = kappa1 / theta`` discussed with Fig. 1 of the article.
        """
        if self.kappa2 is None:
            self.kappa2 = self.kappa1 / self.theta
        assert 1e-4 < self.H <= 0.5

    def approximate_kernel(
        self,
        T: float,
        kernel_rule: KernelRule = KernelRule.EUROPEAN,
        w_bar_max: Optional[float] = None,
    ):
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
        from .kernel import approximate_kernel

        return approximate_kernel(self, T, kernel_rule, w_bar_max)

    def to_dict(self) -> Dict[str, Any]:
        """return the parameters as a plain dict, including the derived kernel arrays."""
        return asdict(self)

    def to_str(self) -> str:
        """return the six model parameters of Eq. (3.12) formatted to two decimals."""
        return (
            f"sigma0={self.sigma0:0.2f}, theta={self.theta:0.2f}, kappa1={self.kappa1:0.2f}, kappa2={self.kappa2:0.2f}, "
            f"beta={self.beta:0.2f}, volvol={self.volvol:0.2f}"
        )

    def set_vol_backbone(self, vol_backbone: pd.Series) -> None:
        """
        attach deterministic price-diffusion loadings eta indexed by time to maturity.

        The volatility state drift and theta are unchanged. The model forward variance is
        eta(t)^2 E[sigma_t^2], so eta^2 alone is not a forward-variance curve.
        """
        self.vol_backbone = vol_backbone

    def get_vol_backbone_eta(self, tau: float) -> float:
        """
        return the backbone scaling eta at the nearest quoted maturity at or beyond tau.

        Returns the core loading 1.0 when no backbone is attached.
        """
        if self.vol_backbone is not None:
            nearest_tau = find_nearest(
                a=self.vol_backbone.index.to_numpy(), value=tau, is_equal_or_largest=True
            )
            vol_backbone_eta = self.vol_backbone.loc[nearest_tau]
        else:
            vol_backbone_eta = 1.0
        return vol_backbone_eta

    def get_vol_backbone_etas(self, ttms: np.ndarray) -> np.ndarray:
        """return :meth:`get_vol_backbone_eta` evaluated over an array of maturities."""
        if self.vol_backbone is not None:
            vol_backbone_etas = np.ones_like(ttms)
            for idx, tau in enumerate(ttms):
                nearest_tau = find_nearest(
                    a=self.vol_backbone.index.to_numpy(), value=tau, is_equal_or_largest=True
                )
                vol_backbone_etas[idx] = self.vol_backbone.loc[nearest_tau]
        else:
            vol_backbone_etas = np.ones_like(ttms)
        return vol_backbone_etas

    @property
    def kappa(self) -> float:
        """
        effective mean-reversion rate kappa = kappa1 + kappa2 theta of Eq. (3.32).

        The rate at which the mean-adjusted volatility Y_t = sigma_t - theta reverts to
        zero in the linearised drift, and the diagonal scale of Lambda^(1,k*) in
        Eq. (3.48).
        """
        return self.kappa1 + self.kappa2 * self.theta

    @property
    def theta2(self) -> float:
        """square of the mean volatility, theta^2."""
        return self.theta * self.theta

    @property
    def vartheta2(self) -> float:
        """
        total instantaneous variance of the volatility process, vartheta^2 = beta^2 + epsilon^2.

        Equation (3.13). The diffusion of Eq. (3.19) is v(sigma) = vartheta sigma, so
        the two Brownian loadings enter the moment ODEs and the affine expansion only
        through this sum; beta enters separately through the measure-dependent terms of
        Eq. (4.3).
        """
        return self.beta * self.beta + self.volvol * self.volvol

    @property
    def vartheta(self) -> float:
        """
        total volatility of volatility vartheta = sqrt(beta^2 + epsilon^2).

        The total volatility-of-volatility argument ``volvol`` of the rough split-step scheme in
        ``rough_logsv.split_simulation``.
        """
        return np.sqrt(self.vartheta2)

    @property
    def rho(self) -> float:
        """
        instantaneous correlation beta / vartheta between the price and volatility Brownian motions.

        The correlation argument ``rho`` of the rough split-step scheme in
        ``rough_logsv.split_simulation``.
        """
        return self.beta / self.vartheta

    @property
    def gamma(self) -> float:
        """
        quadratic mean-reversion rate under the pure quadratic drift, kappa1 / theta.

        The case kappa2 = kappa1 / theta identified with Fig. 1 of the article.
        """
        return self.kappa1 / self.theta

    @property
    def eta(self) -> float:
        """
        exponent eta of the steady-state density of the volatility, Eq. (3.38).

        The generalized inverse Gaussian steady state of Eq. (3.38) is
        G(sigma) = c sigma^(eta - 1) exp{-(q / sigma + b sigma)} with
        eta = 2 (kappa2 theta - kappa1) / vartheta^2 - 1.
        """
        return 2.0 * (self.kappa2 * self.theta - self.kappa1) / self.vartheta2 - 1.0

    def get_x_grid(self, ttm: float = 1.0, n_stdevs: float = 3.0, n: int = 200) -> np.ndarray:
        """
        spatial grid for the density of the log-price X_tau.

        Centred on the drift -0.5 sigma_t^2 of Eq. (3.16) with sigma_t the
        maturity-averaged volatility, and spanning ``n_stdevs + 1`` standard
        deviations either side. Used for the Fourier inversion illustrated in
        Fig. 6(a).
        """
        from .grids import get_x_grid

        return get_x_grid(self, ttm, n_stdevs, n)

    def get_sigma_grid(self, ttm: float = 1.0, n_stdevs: float = 3.0, n: int = 200) -> np.ndarray:
        """
        spatial grid on [0, .] for the density of the volatility sigma_tau.

        Half-width scales with sqrt(vartheta^2 ttm), the diffusive spread of
        Eq. (3.19). Used for Fig. 6(c).
        """
        from .grids import get_sigma_grid

        return get_sigma_grid(self, ttm, n_stdevs, n)

    def get_qvar_grid(self, ttm: float = 1.0, n_stdevs: float = 3.0, n: int = 200) -> np.ndarray:
        """
        spatial grid on [0, .] for the density of the quadratic variance I_tau.

        I_t is defined by dI_t = sigma_t^2 dt in Eq. (3.12). Used for Fig. 6(b).
        """
        from .grids import get_qvar_grid

        return get_qvar_grid(self, ttm, n_stdevs, n)

    def get_variable_space_grid(
        self,
        variable_type: VariableType = VariableType.LOG_RETURN,
        ttm: float = 1.0,
        n_stdevs: float = 3,
        n: int = 200,
    ) -> np.ndarray:
        """
        dispatch to the spatial grid of the state variable selected by variable_type.

        Raises
        ------
        NotImplementedError
            If variable_type is not one of the three state variables of Eq. (3.34).
        """
        from .grids import get_variable_space_grid

        return get_variable_space_grid(self, variable_type, ttm, n_stdevs, n)

    def get_vol_moments_lambda(self, n_terms: int = 4) -> np.ndarray:
        """
        assemble the truncated moment generator Lambda^(1,k*) of Eq. (3.48).

        The mean-adjusted volatility Y_t = sigma_t - theta of Eq. (3.32) has moments
        m_bar^(n)(tau) = E[Y_tau^n] solving the recursion of Eq. (3.50). Truncating at
        order k* under the closure of Eq. (3.51) gives the finite linear system
        ``d_tau M = Lambda M + C`` of Eq. (3.48), whose generator this returns:

            row 1   (-kappa, -kappa2, 0, ...)
            row n   (c(n) theta^2, 2 c(n) theta, c(n) - n kappa, -n kappa2, ...)

        with c(n) = 0.5 vartheta^2 n (n - 1) from Eq. (3.46).

        Parameters
        ----------
        n_terms : int, default 4
            Truncation order k*. Fig. 2 compares k* = 4 and k* = 8 against Monte Carlo;
            k* = 4 matches the first two moments, k* = 8 the first four.

        Returns
        -------
        np.ndarray
            Array with shape ``(n_terms, n_terms)``.
            Lower-Hessenberg generator Lambda^(1,k*).
        """

        from .moments import get_vol_moments_lambda

        return get_vol_moments_lambda(self, n_terms)

    def assert_vol_moments_stability(self, n_terms: int = 4):
        """
        print whether all eigenvalues of Lambda^(1,k*) have negative real part.

        The regularity condition under Eq. (3.50) that makes the truncation of
        Eq. (3.51) admissible. Despite the name this asserts nothing; it prints.
        """
        from .moments import assert_vol_moments_stability

        return assert_vol_moments_stability(self, n_terms)

    def print_vol_moments_stability(self, n_terms: int = 4) -> None:
        """
        print the per-moment diagonal conditions c(n) - n kappa and the spectrum of Lambda.
        """
        from .moments import print_vol_moments_stability

        return print_vol_moments_stability(self, n_terms)
