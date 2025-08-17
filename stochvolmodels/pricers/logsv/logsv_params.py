"""
implementation of log sv params
"""
import numpy as np
import pandas as pd
from numpy import linalg as la
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

from stochvolmodels import VariableType, find_nearest
from stochvolmodels.pricers.model_pricer import ModelParams
from stochvolmodels.pricers.rough_logsv.RoughKernel import european_rule


@dataclass
class LogSvParams(ModelParams):
    """
    Implementation of model params class
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
        if self.kappa2 is None:
            self.kappa2 = self.kappa1 / self.theta
        assert 1e-4 < self.H <= 0.5

    def approximate_kernel(self, T: float, N: int):
        assert 1 <= N <= 5  # not keen to use large N
        if self.H >= 0.4:
            N = N if N<=2 else 2
            self.nodes, self.weights = european_rule(self.H, N, T)
        elif N > 1 and self.H<0.49:
            self.nodes, self.weights = european_rule(self.H, N, T)
        else:
            self.weights = np.array([1.0])
            self.nodes = np.array([1e-3])


    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_str(self) -> str:
        return f"sigma0={self.sigma0:0.2f}, theta={self.theta:0.2f}, kappa1={self.kappa1:0.2f}, kappa2={self.kappa2:0.2f}, " \
               f"beta={self.beta:0.2f}, volvol={self.volvol:0.2f}"

    def set_vol_backbone(self, vol_backbone: pd.Series) -> None:
        self.vol_backbone = vol_backbone

    def get_vol_backbone_eta(self, tau: float) -> float:
        if self.vol_backbone is not None:
            nearest_tau = find_nearest(a=self.vol_backbone.index.to_numpy(), value=tau, is_equal_or_largest=True)
            vol_backbone_eta = self.vol_backbone.loc[nearest_tau]
        else:
            vol_backbone_eta = 1.0
        return vol_backbone_eta

    def get_vol_backbone_etas(self, ttms: np.ndarray) -> np.ndarray:
        if self.vol_backbone is not None:
            vol_backbone_etas = np.ones_like(ttms)
            for idx, tau in enumerate(ttms):
                nearest_tau = find_nearest(a=self.vol_backbone.index.to_numpy(), value=tau, is_equal_or_largest=True)
                vol_backbone_etas[idx] = self.vol_backbone.loc[nearest_tau]
        else:
            vol_backbone_etas = np.ones_like(ttms)
        return vol_backbone_etas

    @property
    def kappa(self) -> float:
        return self.kappa1+self.kappa2*self.theta

    @property
    def theta2(self) -> float:
        return self.theta*self.theta

    @property
    def vartheta2(self) -> float:
        return self.beta*self.beta + self.volvol*self.volvol

    @property
    def gamma(self) -> float:
        """
        assume kappa2 = kappa1 / theta
        """
        return self.kappa1 / self.theta

    @property
    def eta(self) -> float:
        """
        assume kappa2 = kappa1 / theta
        """
        return self.kappa1 * self.theta / self.vartheta2 - 1.0

    def get_x_grid(self, ttm: float = 1.0, n_stdevs: float = 3.0, n: int = 200) -> np.ndarray:
        """
        spacial grid to compute density of x
        """
        sigma_t = np.sqrt(ttm * 0.5 * (np.square(self.sigma0) + np.square(self.theta)))
        drift = - 0.5*sigma_t*sigma_t
        stdev = (n_stdevs+1)*sigma_t
        return np.linspace(-stdev+drift, stdev+drift, n)

    def get_sigma_grid(self, ttm: float = 1.0, n_stdevs: float = 3.0, n: int = 200) -> np.ndarray:
        """
        spacial grid to compute density of sigma
        """
        sigma_t = np.sqrt(0.5*(np.square(self.sigma0) + np.square(self.theta)))
        vvol = 0.5*np.sqrt(self.vartheta2*ttm)
        return np.linspace(0.0, sigma_t+n_stdevs*vvol, n)

    def get_qvar_grid(self, ttm: float = 1.0, n_stdevs: float = 3.0, n: int = 200) -> np.ndarray:
        """
        spacial grid to compute density of i
        """
        sigma_t = np.sqrt(ttm * (np.square(self.sigma0) + np.square(self.theta)))
        vvol = np.sqrt(self.vartheta2)*ttm
        return np.linspace(0.0, sigma_t+n_stdevs*vvol, n)

    def get_variable_space_grid(self, variable_type: VariableType = VariableType.LOG_RETURN,
                                ttm: float = 1.0,
                                n_stdevs: float = 3,
                                n: int = 200
                                ) -> np.ndarray:
        if variable_type == VariableType.LOG_RETURN:
            return self.get_x_grid(ttm=ttm, n_stdevs=n_stdevs, n=n)
        if variable_type == VariableType.SIGMA:
            return self.get_sigma_grid(ttm=ttm, n_stdevs=n_stdevs, n=n)
        elif variable_type == VariableType.Q_VAR:
            return self.get_qvar_grid(ttm=ttm, n_stdevs=n_stdevs, n=n)
        else:
            raise NotImplementedError

    def get_vol_moments_lambda(self,
                               n_terms: int = 4
                               ) -> np.ndarray:

        kappa2 = self.kappa2
        kappa = self.kappa
        vartheta2 = self.vartheta2
        theta = self.theta
        theta2 = self.theta2

        def c(n: int) -> float:
            return 0.5 * vartheta2 * n * (n - 1.0)

        lambda_m = np.zeros((n_terms, n_terms))
        lambda_m[0, 0] = -kappa
        lambda_m[0, 1] = -kappa2
        lambda_m[1, 0] = 2.0*c(2) * theta
        lambda_m[1, 1] = c(2) - 2.0*kappa
        lambda_m[1, 2] = -2.0*kappa2

        for n_ in np.arange(2, n_terms):
            n = n_ + 1  # n_ is array counter, n is formula counter
            c_n = c(n)
            lambda_m[n_, n_ - 2] = c_n * theta2
            lambda_m[n_, n_ - 1] = 2.0 * c_n * theta
            lambda_m[n_, n_] = c_n - n*kappa
            if n_ + 1 < n_terms:
                lambda_m[n_, n_ + 1] = -n*kappa2

        return lambda_m

    def assert_vol_moments_stability(self, n_terms: int = 4):
        lambda_m = self.get_vol_moments_lambda(n_terms=n_terms)
        w, v = la.eig(lambda_m)
        cond = np.all(np.real(w)<0.0)
        print(f"vol moments stable = {cond}")

    def print_vol_moments_stability(self, n_terms: int = 4) -> None:
        def c(n: int) -> float:
            return 0.5 * self.vartheta2 * n * (n - 1.0)

        cond_m2 = c(2) - 2.0*self.kappa
        print(f"con2:\n{cond_m2}")
        cond_m3 = c(3) - 3.0*self.kappa
        print(f"con3:\n{cond_m3}")
        cond_m4 = c(4) - 4.0*self.kappa
        print(f"cond4:\n{cond_m4}")

        lambda_m = self.get_vol_moments_lambda(n_terms=n_terms)
        print(f"lambda_m:\n{lambda_m}")

        w, v = la.eig(lambda_m)
        print(f"eigenvalues w:\n{w}")
        print(f"vol moments stable = {np.all(np.real(w)<0.0)}")
