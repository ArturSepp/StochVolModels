"""
data class implementation for mc params
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy import linalg as sla
from numpy import linalg as la
from generic.config import VariableType
from generic.model_pricer import ModelParams


@dataclass
class LogSvParams(ModelParams):
    sigma0: float = 0.2
    theta: float = 0.2
    kappa1: float = 1.0
    kappa2: Optional[float] = 2.5  # Optional is mapped to self.kappa1 / self.theta
    beta: float = -1.0
    volvol: float = 1.0

    def __post_init__(self):
        if self.kappa2 is None:
            self.kappa2 = self.kappa1 / self.theta

    def to_dict(self):
        return dict(sigma0=self.sigma0, theta=self.theta, kappa1=self.kappa1, kappa2=self.kappa2, beta=self.beta, volvol=self.volvol)

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

    def get_x_grid(self, ttm: float = 1.0, n_stdevs: int = 3, n: int = 200) -> np.ndarray:
        sigma_t = np.sqrt(ttm * 0.5 * (np.square(self.sigma0) + np.square(self.theta)))
        drift = - 0.5*sigma_t*sigma_t
        stdev = (n_stdevs+1)*sigma_t
        return np.linspace(-stdev+drift, stdev+drift, n)

    def get_sigma_grid(self, ttm: float = 1.0, n_stdevs: int = 3, n: int = 200) -> np.ndarray:
        sigma_t = np.sqrt(ttm * 0.5 * (np.square(self.sigma0) + np.square(self.theta)))
        vvol = np.sqrt(self.vartheta2/np.abs(2.0*self.kappa1))
        return np.linspace(0.0, sigma_t+n_stdevs*vvol, n)

    def get_qvar_grid(self, ttm: float = 1.0, n_stdevs: int = 3, n: int = 200) -> np.ndarray:
        sigma_t = np.sqrt(ttm * 0.5 * (np.square(self.sigma0) + np.square(self.theta)))
        vvol = np.sqrt(self.vartheta2/np.abs(2.0*self.kappa1))
        return np.linspace(0.0, sigma_t+n_stdevs*vvol, n)

    def get_variable_space_grid(self, variable_type: VariableType = VariableType.LOG_RETURN,
                                ttm: float = 1.0, n_stdevs: int = 3, n: int = 200
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

        # e_m = sla.expm(lambda_m)
        # print(f"e_m:\n{e_m}")
        # e_inv = la.inv(lambda_m)
        # print(f"e_inv:\n{e_inv}")
        # m = np.dot(e_inv, (e_m - np.eye(n_terms)))
        # print(f"m:\n{m}")
