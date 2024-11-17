import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from numba.typed import List
from abc import ABC, abstractmethod

from stochvolmodels.pricers.factor_hjm.rate_core import bond, swap_grad


class BasisHJM(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_basis(self, tau: float) -> np.ndarray:
        """return (functional) main basis for a given tenor tau"""
        pass

    @abstractmethod
    def get_aux_basis(self, tau: float) -> np.ndarray:
        """return (functional) auxiliary basis for a given tenor tau"""
        pass

    @staticmethod
    @abstractmethod
    def get_nb_factors(cls) -> np.ndarray:
        """return number of model factors """
        pass

    @staticmethod
    @abstractmethod
    def get_nb_aux_factors(cls) -> np.ndarray:
        """return number of auxiliary factors """
        pass

    @abstractmethod
    def bond_coeffs(self, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def calc_Omega(self, M: np.ndarray) -> np.ndarray:
        pass

    # @njit(cache=False, fastmath=True) # TODO: cannot make it numba as it is member function
    def _bond(self, nb_factors: int, nb_aux_factors: int,
              t: float, T: float,
              x: np.ndarray, y: np.ndarray,
              ccy: str,
              m: int = 0) -> np.ndarray:
        assert t <= T
        # sime of x is # simulations x #factors
        # assume that in matrix X the row is for trials
        # column is for factors
        assert x.shape[-1] == nb_factors and y.shape[-1] == nb_aux_factors
        tau = T - t
        B_PX, B_PY = self.bond_coeffs(tau)
        # there are some other checks at bond function
        bond_value = bond(t, T, x, y, B_PX, B_PY, ccy, m)

        return bond_value

    def _get_matrix_B(self, nb_factors: int, key_terms: np.ndarray) -> np.ndarray:
        nb_key_terms = key_terms.size
        B = np.zeros((nb_key_terms, nb_factors))
        for idx, tau in enumerate(key_terms):
            B[idx, :] = 1.0/tau * self.bond_coeffs(tau)[0]

        return B

    # @njit(cache=False, fastmath=True) # TODO: cannot make it numba as it is member function
    def annuity(self, t: float, ts_sw: np.ndarray,
                x: np.ndarray, y: np.ndarray,
                ccy: str,
                m: int = 0) -> np.ndarray:
        ann = 0
        for i in range(1, ts_sw.size):
            bond_value = self.bond(t, ts_sw[i], x, y, ccy, m)
            ann = ann + (ts_sw[i] - ts_sw[i - 1]) * bond_value
        return ann

    # @njit(cache=False, fastmath=True) # TODO: cannot make it numba as it is member function
    def swap_rate(self, t: float, ts_sw: np.ndarray,
                  x: np.ndarray, y: np.ndarray,
                  ccy: str) -> (np.ndarray, np.ndarray):
        denumer0 = 0
        denumer1 = 0
        for i in range(1, ts_sw.size):
            denumer0 = denumer0 + (ts_sw[i] - ts_sw[i - 1]) * self.bond(t, ts_sw[i], x, y, ccy=ccy, m=0)
            denumer1 = denumer1 + (ts_sw[i] - ts_sw[i - 1]) * self.bond(t, ts_sw[i], x, y, ccy=ccy, m=1)

        numer0 = self.bond(t, ts_sw[0], x, y, ccy=ccy, m=0) - self.bond(t, ts_sw[-1], x, y, ccy=ccy, m=0)
        numer1 = self.bond(t, ts_sw[0], x, y, ccy=ccy, m=1) - self.bond(t, ts_sw[-1], x, y, ccy=ccy, m=1)
        # ad-hoc fix
        # if numer1.ndim == 2:
        #     numer1 = numer1[:, 0]

        value0 = numer0 / denumer0
        value1 = swap_grad(numer0=numer0, numer1=numer1, denumer0=denumer0, denumer1=denumer1)

        return value0, value1

    # @njit(cache=False, fastmath=True) # TODO: cannot make it numba as it is member function
    def libor_rate(self, t: float, t_start: float, t_end: float,
                   x: np.ndarray, y: np.ndarray,
                   ccy: str):
        zcb_start = self.bond(t, t_start, x, y, ccy=ccy, m=0)
        zcb_end = self.bond(t, t_end, x, y, ccy=ccy, m=0)
        libor = 1.0 / (t_end - t_start) * (zcb_start / zcb_end - 1.0)

        return libor

    # @njit(cache=False, fastmath=True) # TODO: cannot make it numba as it is member function
    def calculate_swap_rate(self,
                            ttm: float,
                            x0: np.ndarray,
                            y0: np.ndarray,
                            I0: np.ndarray,
                            ts_sw: np.ndarray,
                            ccy: str) -> List[np.ndarray, ...]:
        # calculate swap rates using simulated states
        # ts_sw = get_default_swap_term_structure(ttm)
        s_mc = self.swap_rate(t=ttm, ts_sw=ts_sw, x=x0, y=y0, ccy=ccy)[0]
        ann_mc = self.annuity(t=ttm, ts_sw=ts_sw, x=x0, y=y0, m=0, ccy=ccy)
        numer = 1.0 / self.bond(t=0, T=ttm, x=np.zeros((1, x0.shape[1])), y=np.zeros((1, y0.shape[1])), m=0, ccy=ccy) * np.exp(I0)
        return s_mc, ann_mc, numer


#############################################################################
#                     Single factor Cheyette model
#############################################################################
@dataclass
class Cheyette1D(BasisHJM):
    meanrev: float

    def __post_init__(self):
        assert self.meanrev > 0
        self.nb_factors = Cheyette1D.get_nb_factors()
        self.nb_aux_factors = Cheyette1D.get_nb_aux_factors()

    def get_basis(self, tau: float) -> np.ndarray:
        raise NotImplementedError(f"not supported for Cheyette1D")

    def get_aux_basis(self, tau: float):
        raise NotImplementedError(f"not supported for Cheyette1D")

    @classmethod
    def get_nb_factors(cls) -> int:
        return 1

    @classmethod
    def get_nb_aux_factors(cls) -> int:
        return 1

    def get_generating_matrix(self) -> np.ndarray:
        # D = np.zeros((self.nb_factors, self.nb_factors))
        # D[1, 1] = D[2, 2] = -self.meanrev
        # D[1, 2] = 1.0
        # return D
        raise NotImplementedError(f"not supported for Cheyette1D")

    def get_aux_generating_matrix(self) -> np.ndarray:
        # D = np.zeros((self.nb_aux_factors, self.nb_aux_factors))
        # D[0, 1] = 1.0
        # D[2, 2] = D[3, 3] = D[4, 4] = -self.meanrev
        # D[2, 3] = D[3, 4] = 1.0
        # D[5, 5] = D[6, 6] = D[7, 7] = -2.0 * self.meanrev
        # D[5, 6] = D[6, 7] = 1.0
        # return D
        raise NotImplementedError(f"not supported for Cheyette1D")

    def get_matrix_B(self, key_terms: List[float]) -> np.ndarray:
        # assert len(key_terms) == 3  # for now we assume three key terms
        # B = np.zeros((len(key_terms), self.nb_factors))
        # for idx, tau in enumerate(key_terms):
        #     B[idx, :] = self.get_basis(tau)
        #
        # return B
        raise NotImplementedError(f"not supported for Cheyette1D")

    def calc_Omega(self, M: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"not supported for Cheyette1D")

    # @njit(cache=False, fastmath=True) # TODO: cannot make it numba as it is member function
    def bond_coeffs(self, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        G_tau = (1.0 - np.exp(-self.meanrev * tau)) / self.meanrev
        B_PX = np.array([G_tau])
        B_PY = np.array([0.5*G_tau*G_tau])
        return B_PX, B_PY

    # @njit(cache=False, fastmath=True) # TODO: cannot make it numba as it is member function
    def bond(self, t: float, T: float,
             x: np.ndarray, y: np.ndarray,
             ccy: str,
             m: int = 0) -> np.ndarray:
        assert t <= T
        if x.ndim == 1:
            # sime of x is #factors * # simulations
            # assume that in matrix x the columns is for trials
            # rows are for factors
            assert x.shape[0] == y.shape[0]
        tau = T - t
        B_PX, B_PY = self.bond_coeffs(tau)
        # there are some other checks at bond function
        bond_value = bond(t, T, x, y, B_PX, B_PY, ccy, m)
        if x.ndim == 0:
            bond_value = bond_value[0][0]

        return bond_value


#############################################################################
#                     Nelson-Siegel FHJM model
#############################################################################
@dataclass
class NelsonSiegel(BasisHJM):
    meanrev: float
    key_terms: np.ndarray

    def __post_init__(self):
        assert self.meanrev > 0
        self.nb_factors = NelsonSiegel.get_nb_factors()
        self.nb_aux_factors = NelsonSiegel.get_nb_aux_factors()
        # same number of key terms and factors
        assert self.key_terms.size == self.nb_factors


    def get_basis(self, tau: float) -> np.ndarray:
        mrv = self.meanrev
        exp_mrv = np.exp(-mrv * tau)
        return np.array([1.0, exp_mrv, tau * exp_mrv])

    def get_aux_basis(self, tau: float):
        mrv = self.meanrev
        exp_mrv = np.exp(-mrv * tau)
        exp_mrv2 = np.exp(-2.0 * mrv * tau)
        return np.array([1.0, tau,
                         exp_mrv, tau * exp_mrv, 0.5 * tau * tau * exp_mrv,
                         exp_mrv2, tau * exp_mrv2, 0.5 * tau * tau * exp_mrv2])

    @classmethod
    def get_nb_factors(cls) -> int:
        return 3

    @classmethod
    def get_nb_aux_factors(cls) -> int:
        return 8

    def get_generating_matrix(self) -> np.ndarray:
        D = np.zeros((self.nb_factors, self.nb_factors))
        D[1, 1] = D[2, 2] = -self.meanrev
        D[1, 2] = 1.0
        return D

    def get_aux_generating_matrix(self) -> np.ndarray:
        D = np.zeros((self.nb_aux_factors, self.nb_aux_factors))
        D[0, 1] = 1.0
        D[2, 2] = D[3, 3] = D[4, 4] = -self.meanrev
        D[2, 3] = D[3, 4] = 1.0
        D[5, 5] = D[6, 6] = D[7, 7] = -2.0 * self.meanrev
        D[5, 6] = D[6, 7] = 1.0
        return D


    def get_matrix_B(self) -> np.ndarray:
        return self._get_matrix_B(self.nb_factors, self.key_terms)

    def calc_Omega(self, M: np.ndarray) -> np.ndarray:
        assert M.shape == (self.nb_factors, self.nb_factors)
        mrv = self.meanrev
        mrv2 = mrv * mrv
        mrv3 = mrv * mrv2
        Omega = np.zeros((self.nb_aux_factors,))
        Omega[0] = M[0, 1] / mrv + M[0, 2] / mrv2
        Omega[1] = M[0, 0]
        Omega[2] = -M[0, 1] / mrv - M[0, 2] / mrv2 + M[1, 1] / mrv + M[1, 2] / mrv2
        Omega[3] = M[0, 1] - M[0, 2] / mrv + M[1, 2] / mrv + M[2, 2] / mrv2
        Omega[4] = 2.0 * M[0, 2]
        Omega[5] = -M[1, 1] / mrv - M[1, 2] / mrv2
        Omega[6] = -2.0 / mrv * M[1, 2] - 1.0 / mrv2 * M[2, 2]
        Omega[7] = -2.0 / mrv * M[2, 2]

        return Omega

    def bond(self, t: float, T: float,
               x: np.ndarray, y: np.ndarray,
               ccy: str,
               m: int = 0) -> np.ndarray:
        return self._bond(self.nb_factors, self.nb_aux_factors, t, T, x, y, ccy, m)

    # @njit(cache=False, fastmath=True) # TODO: cannot make it numba as it is member function
    def bond_coeffs(self, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        mrv = self.meanrev
        mrv2 = mrv * mrv
        mrv3 = mrv * mrv * mrv
        mt = mrv * tau
        mt2 = mt * mt
        exp_mrv = np.exp(-mt)
        exp_mrv2 = np.exp(-2.0 * mt)
        B_PX = np.array([tau, (1.0 - exp_mrv) / mrv, (1.0 - exp_mrv * (1.0 + mt)) / mrv2])
        B_PY = np.array([tau, 0.5 * tau * tau,
                         (1.0 - exp_mrv) / mrv, (1.0 - exp_mrv * (1.0 + mt)) / mrv2,
                         (1.0 - exp_mrv * (1.0 + mt + 0.5 * mt2)) / mrv3,
                         0.5 * (1.0 - exp_mrv2) / mrv, 0.25 * (1.0 - exp_mrv2 * (1 + 2.0 * mt))/mrv2,
                         0.125 * (1.0 - exp_mrv2 * (1.0 + 2.0 * mt + 2.0 * mt2)) / mrv3])
        return B_PX, B_PY


#############################################################################
#                     Multi-factor Pure Exponential FHJM model
#############################################################################
@dataclass
class CheyettePEND(BasisHJM):
    mrv0: float
    mrv_delta: float
    key_terms: np.ndarray

    def __post_init__(self):
        assert self.mrv0 > 0
        assert self.mrv_delta > 0
        self.nb_factors = CheyettePEND.get_nb_factors()
        self.nb_aux_factors = CheyettePEND.get_nb_aux_factors()
        # same number of key terms and factors
        assert self.key_terms.size == self.nb_factors

    def calc_mrvs(self):
        mrvs = np.arange(self.mrv0, self.mrv0 + self.mrv_delta * self.nb_factors - 1e-6, self.mrv_delta)
        return mrvs

    def calc_mrvs_extra(self):
        mrvs_extra = np.arange(2.0*self.mrv0, 2.0*self.mrv0 + self.mrv_delta * (2.0 * self.nb_factors - 2.0) + 1e-6, self.mrv_delta)
        return mrvs_extra

    def get_basis(self, tau: float) -> np.ndarray:
        mrvs = self.calc_mrvs()
        exp_mrv = np.exp(-mrvs * tau)
        return exp_mrv

    def get_aux_basis(self, tau: float):
        mrvs = self.calc_mrvs()
        exp_mrvs = np.exp(-mrvs * tau)
        # when lambda_1,...,lambda_d is evenly spaced, then
        # lambda_i + lambda_j is evenly spaced from alpha to alpha + (i+j-2)*delta
        # where alpha = 2*lambda_1
        mrvs_extra = self.calc_mrvs_extra()
        exp_mrvs_extra = np.exp(-mrvs_extra * tau)
        return np.concatenate((exp_mrvs, exp_mrvs_extra))

    @classmethod
    def get_nb_factors(cls) -> int:
        return 3

    @classmethod
    def get_nb_aux_factors(cls) -> int:
        d = cls.get_nb_factors()
        return d + 2*d - 1

    def get_generating_matrix(self) -> np.ndarray:
        D = -np.diag(self.calc_mrvs())
        assert D.shape == (self.nb_factors, self.nb_factors)
        return D

    def get_aux_generating_matrix(self) -> np.ndarray:
        D = -np.diag(np.concatenate((self.calc_mrvs(), self.calc_mrvs_extra())))
        assert D.shape == (self.nb_aux_factors, self.nb_aux_factors)
        return D

    def get_matrix_B(self) -> np.ndarray:
        return self._get_matrix_B(self.nb_factors, self.key_terms)

    def calc_Omega(self, M: np.ndarray) -> np.ndarray:
        assert M.shape == (self.nb_factors, self.nb_factors)
        mrvs = self.calc_mrvs()
        mrvs_extra = self.calc_mrvs_extra()
        assert self.nb_aux_factors == mrvs.size + mrvs_extra.size

        Omega = np.zeros((self.nb_aux_factors,))
        for i in range(mrvs.size):
            Omega[i] = np.dot(M[i,:], 1.0/mrvs)
        for k in range(mrvs_extra.size):
            sum_fix_k = 0.0
            for i,j in zip(range(k,-1,-1), range(0,k+1,1)):
                if 0 <= i < self.nb_factors and 0 <= j < self.nb_factors:
                    sum_fix_k = sum_fix_k - M[i,j]/mrvs[j]
            # assign
            Omega[mrvs.size+k] = sum_fix_k
        return Omega

    def bond(self, t: float, T: float,
             x: np.ndarray, y: np.ndarray,
             ccy: str,
             m: int = 0) -> np.ndarray:
        return self._bond(self.nb_factors, self.nb_aux_factors, t, T, x, y, ccy, m)

    # @njit(cache=False, fastmath=True) # TODO: cannot make it numba as it is member function
    def bond_coeffs(self, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        mrvs = self.calc_mrvs()
        exp_mrvs = np.exp(-mrvs * tau)
        mrvs_extra = self.calc_mrvs_extra()
        exp_mrvs_extra = np.exp(-mrvs_extra * tau)
        B_PX = np.divide(1.0 - exp_mrvs, mrvs)
        B_PY = np.concatenate((np.divide(1.0 - exp_mrvs, mrvs), np.divide(1.0 - exp_mrvs_extra, mrvs_extra)))
        return B_PX, B_PY


