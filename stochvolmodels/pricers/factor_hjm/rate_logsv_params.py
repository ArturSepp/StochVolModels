import numpy as np
from typing import Union, Tuple
from dataclasses import dataclass
from numba.typed import List
from scipy.integrate import solve_ivp

from stochvolmodels.pricers.model_pricer import ModelParams
from stochvolmodels.pricers.factor_hjm.rate_factor_basis import Cheyette1D, NelsonSiegel, CheyettePEND
from stochvolmodels.pricers.factor_hjm.rate_core import pw_const, G, bracket, get_default_swap_term_structure, \
    generate_ttms_grid


@dataclass
class TermStructure:
    ts: np.ndarray
    xs: np.ndarray
    flat_extrapol: bool = False

    def __post_init__(self):
        """
        check that market data dimensions are consistent
        """
        if self.ts.ndim != 1:
            raise ValueError('ts must have 1 dimension')
        if self.xs.ndim != 1 and self.xs.ndim != 2:
            raise ValueError('xs must have dimension of one or two')
        if self.ts.shape[0] - 1 != self.xs.shape[0]:
            raise ValueError('abcsissas and ordinates must have same shape')


    # @njit(cache=False, fastmath=True) # TODO: cannot make it numba as it is member function
    def pw_const(self, t: float) -> float:
        """
        Piecewise constant interpolation with possibly flat extrapolation
        """
        return pw_const(self.ts, self.xs, t, self.flat_extrapol, shift=1)

    # @njit(cache=False, fastmath=True) # TODO: cannot make it numba as it is member function
    def interpolate(self, times: np.ndarray) -> np.ndarray:
        interp_values = np.ones_like(times) * np.nan
        for idx, t in enumerate(times):
            interp_values[idx] = self.pw_const(t)
        return interp_values

    @classmethod
    def create_from_scalar(cls, ts: np.ndarray, xs: float, flat_extrapol: bool = False):
        return TermStructure(ts=ts, xs=np.ones_like(ts[1:]) * xs, flat_extrapol=flat_extrapol)

    @classmethod
    def create_multi_fact_from_vec(cls, ts: np.ndarray, xs: np.ndarray, flat_extrapol: bool = False):
        assert xs.ndim == 1
        # when value is 1D vector, its value are for each factor
        # we have as many columns as factors
        # as many rows as time points
        xs_ = np.zeros((ts[1:].size, xs.size))
        for j, val in enumerate(xs):
            xs_[:, j] = val * np.ones_like(ts[1:])
        return TermStructure(ts=ts, xs=xs_, flat_extrapol=flat_extrapol)



@dataclass
class RateLogSvParamsBase(ModelParams):
    """
    Implementation of model params class
    """
    sigma0: float
    theta: float
    kappa1: float
    kappa2: float
    alpha: TermStructure  # term-structure
    b: TermStructure  # term-structure
    beta: TermStructure  # term-structure
    volvol: TermStructure  # term-structure
    ccy: str    # currency in discounting
    basis: Cheyette1D  # basis for 1D Cheyette
    term: float  # key swap tenor
    q: float = None


@dataclass
class RateLogSvParams(RateLogSvParamsBase):
    def calc_mean_states(self,
                         expiry: float,
                         t_grid: np.ndarray) -> (np.ndarray, np.ndarray):
        mrv_r = self.basis.meanrev
        ts_sw = get_default_swap_term_structure(expiry=expiry, tenor=self.term)

        def func_mean_rhs(t: float,
                          arg: np.ndarray,
                          kappa_r: float,
                          ts_sw_: np.ndarray,
                          a: TermStructure,
                          beta: TermStructure,
                          kappa1: float,
                          kappa2: float,
                          theta: float) -> np.ndarray:
            res = np.zeros((3,))
            x, y, sigma = arg[0], arg[1], arg[2]
            a_t = self.alpha.pw_const(t)
            beta_t = self.beta.pw_const(t)
            loga_der = 1.0 / self.basis.annuity(t, ts_sw_, x, y, ccy=self.ccy, m=0) * self.basis.annuity(t, ts_sw_, x, y, ccy=self.ccy, m=1)
            res[0] = y - kappa_r * x + loga_der * a_t ** 2 * sigma ** 2
            res[1] = a_t ** 2 * sigma ** 2 - 2.0 * kappa_r * y
            res[2] = (kappa1 + kappa2 * sigma) * (theta - sigma) + a_t * beta_t * loga_der * sigma ** 2

            return res

        A_t0 = np.array([0.0, 0.0, self.sigma0])  # TODO: review initial conditions for x,y
        # solve ODE with initial conditions x(0), y(0)
        oderesult = solve_ivp(fun=func_mean_rhs, t_span=(0, expiry), t_eval=t_grid, y0=A_t0,
                              args=(mrv_r, ts_sw, self.alpha, self.beta, self.kappa1, self.kappa2, self.theta),
                              dense_output=False)
        (mx, my) = oderesult.y[:2, :]
        return mx, my

    def transform_QA_params(self,
                            expiry: float,
                            tenor: float,
                            t_grid: np.ndarray):

        theta = self.theta
        kappa1 = self.kappa1
        kappa2 = self.kappa2
        if self.q is None:
            self.q = self.theta
        q = self.q

        # in fact, tenor in argument is just dummy variable
        # but still need to check for consistency
        assert tenor == self.term
        ts_sw = get_default_swap_term_structure(expiry=expiry, tenor=tenor)
        if expiry not in t_grid:
            raise ValueError(f"expiry must be in grid")
        idx_ttm = np.where(t_grid == expiry)[0][0]
        t_grid = t_grid[:idx_ttm + 1]

        mx_grid, my_grid = self.calc_mean_states(expiry, t_grid)

        swap_der1 = np.ones_like(t_grid)
        ann = np.ones_like(t_grid)
        ann_der1 = np.ones_like(t_grid)
        for idx, (t, mx, my) in enumerate(zip(t_grid, mx_grid, my_grid)):
            swap_der1[idx] = self.basis.swap_rate(t, ts_sw, mx, my, ccy=self.ccy)[1]
            ann[idx] = self.basis.annuity(t, ts_sw, mx, my, ccy=self.ccy, m=0)
            ann_der1[idx] = self.basis.annuity(t, ts_sw, mx, my, ccy=self.ccy, m=1)

        loga_der = ann_der1 / ann

        # interpolate a, beta, vol-vol from coarse grid to finer one
        alpha_interp = self.alpha.interpolate(t_grid)
        beta_interp = self.beta.interpolate(t_grid)
        volvol_interp = self.volvol.interpolate(t_grid)

        a = alpha_interp * swap_der1  # coordinate-wise product of arrays
        beta2 = beta_interp * loga_der  # coordinate-wise product of arrays

        term0 = alpha_interp * beta2 * (q ** 2) + (theta - q) * kappa1 + (theta - q) * kappa2 * q
        term1 = kappa1 - kappa2 * q + 2.0 * (kappa2 - alpha_interp * beta2) * q - (theta - q) * kappa2
        term2 = kappa2 - alpha_interp * beta2

        return a, term0, term1, term2, beta_interp, volvol_interp, ts_sw

    def reduce(self, idx: int):
        param = RateLogSvParams(sigma0=self.sigma0,
                                theta=self.theta,
                                kappa1=self.kappa1,
                                kappa2=self.kappa2,
                                alpha=TermStructure(self.alpha.ts[:idx + 1], self.alpha.xs[:idx]),
                                b=TermStructure(self.b.ts[:idx + 1], self.b.xs[:idx]),
                                beta=TermStructure(self.beta.ts[:idx + 1], self.beta.xs[:idx]),
                                volvol=TermStructure(self.volvol.ts[:idx + 1], self.volvol.xs[:idx]),
                                ccy=self.ccy,
                                basis=self.basis,
                                term=self.term)
        return param


    def transform_QT_params(self,
                            expiry: float,
                            t_start: float,
                            t_end: float,
                            t_grid: np.ndarray):
        theta = self.theta
        kappa1 = self.kappa1
        kappa2 = self.kappa2
        self.q = self.theta
        q = self.q

        # interpolate a, beta, vol-vol from coarse grid to finer one
        alpha_interp = self.alpha.interpolate(t_grid)
        beta_interp = self.beta.interpolate(t_grid)
        volvol_interp = self.volvol.interpolate(t_grid)

        k = self.basis.meanrev
        G_t_T = G(k, t_grid, expiry)
        G_start_end = G(k, t_start, t_end)


        a = alpha_interp * G_start_end * np.exp(-k * (t_start - t_grid))  # coordinate-wise product of arrays
        eta = alpha_interp * G_t_T  # coordinate-wise product of arrays
        beta2 = beta_interp * G_t_T  # coordinate-wise product of arrays
        delta = a * eta  # coordinate-wise product of arrays

        term0 = alpha_interp * beta2 * (q ** 2)
        term1 = kappa1 - kappa2 * q + 2.0 * (kappa2 + alpha_interp * beta2) * q
        term2 = kappa2 + alpha_interp * beta2

        return a, delta, term0, term1, term2, beta_interp, volvol_interp


@dataclass
class MultiFactRateLogSvParams(ModelParams):
    """Implementation of model params class for multi-factor model"""
    sigma0: float
    theta: float
    kappa1: float
    kappa2: float
    beta: TermStructure  # beta is term-structure for each driver X_i
    volvol: TermStructure  # residual vol-vol is same for all factors
    # volatilities are term-structure as A = A_t
    A: np.ndarray  # (d,) -- vector matrix of volatilities of key-term forward rates
    # correlations are flat in time
    R: np.ndarray  # (d*d) -- posdef matrix of correlations of key-term forward rates
    basis: Union[NelsonSiegel, CheyettePEND]  # Nelson-Siegel parametrization
    # B: np.ndarray  # (d*d) -- matrix where rows are basis functions
    ccy: str
    vol_interpolation: str = "DIRECT"
    q: float = None

    @classmethod
    def make_A_2d(cls, A: np.ndarray, ts: np.ndarray):
        if A.ndim == 1:
            A_ = np.zeros((ts.size - 1, A.shape[0]))
            for idx, t in enumerate(A_):
                A_[idx, :] = A
        elif A.ndim == 2:
            A_ = A
        else:
            raise NotImplementedError
        return A_

    def __post_init__(self):
        self.key_terms = self.basis.key_terms
        # term-structure times for beta and vol-vol must be consistent
        assert np.all(self.beta.ts == self.volvol.ts)
        # if A is 2D => make is 3D
        self.A = MultiFactRateLogSvParams.make_A_2d(self.A, self.beta.ts)
        assert self.A.shape[0] == self.beta.ts.size - 1  # exclude t=0
        # number of terms must be same as number of factors, equal to 3
        assert len(self.key_terms) == self.basis.nb_factors
        # number of factors in multi-factor term-structure = number of factors in Nelson-Siegel
        assert self.beta.xs.shape[1] == self.basis.nb_factors
        assert self.A.shape[1] == self.basis.nb_factors

        if self.vol_interpolation not in ["BY_YIELD", "DIRECT"]:
            raise NotImplementedError(f"Wrong vol interpolation type")

        # TODO: check if definition of cholesky decomposion is same assume,
        # TODO: in numpy resuling R is lower triangular

        C = np.zeros((self.A.shape[0], self.A.shape[1], self.A.shape[1]))
        M = np.zeros((self.A.shape[0], self.A.shape[1], self.A.shape[1]))
        Omega = np.zeros((self.beta.ts.size - 1, self.basis.nb_aux_factors))
        # interpolation of the volatility matrix is tedious
        for idx, Ai in enumerate(self.A):
            # if Ai.ndim == 1:
            #     Ai = np.reshape(Ai, (Ai.shape[0], 1))
            if self.vol_interpolation == "BY_YIELD":
                Ci = self.calc_factor_vols(Ai)
            else:
                raise NotImplementedError
            Mi = np.dot(Ci, np.transpose(Ci))
            Omegai = self.basis.calc_Omega(Mi)
            C[idx, :] = Ci
            M[idx, :] = Mi
            Omega[idx, :] = Omegai
        self.C = C
        self.M = M
        self.Omega = Omega
        self.ts = self.beta.ts

    #############################################################################
    #                     Analytics under annuity measure Q^A for swaptions
    #############################################################################
    def calc_QA_mean_states(self,
                            expiry: float,
                            tenor: float,
                            t_grid: np.ndarray,
                            x0: np.ndarray,
                            y0: np.ndarray
                            ) -> (np.ndarray, np.ndarray):
        """Calculate Q^A mean \\bar X_t and \\bar Y_t for multi-factor model at times t_grid"""
        ts_sw = get_default_swap_term_structure(expiry=expiry, tenor=tenor)

        def func_mean_rhs(t: float,
                          arg: np.ndarray,
                          ts_sw_: np.ndarray,
                          kappa1: float,
                          kappa2: float,
                          theta: float) -> np.ndarray:
            # evolve vector \\bar X, vector \\barY and scalar vol
            sz_X = self.basis.nb_factors
            sz_Y = self.basis.nb_aux_factors
            sz = sz_X + sz_Y + 1
            res = np.zeros((sz,))
            x, y, sigma = arg[:sz_X], arg[sz_X:sz_X + sz_Y], arg[-1]
            D_X = self.basis.get_generating_matrix()
            D_Y = self.basis.get_aux_generating_matrix()
            idx_t = bracket(self.ts[1:], t, False)
            M_t = self.M[idx_t]
            Omega_t = self.Omega[idx_t]
            beta_t = self.beta.pw_const(t)
            C_t = self.C[idx_t]
            loga_der = 1.0 / self.basis.annuity(t, ts_sw_, x, y, self.ccy, 0) * self.basis.annuity(t, ts_sw_, x, y,
                                                                                                   self.ccy, 1)[0,:]
            # M_t must be matrix, L_X must be a vector column
            res[:sz_X] = np.dot(D_X, x) + sigma ** 2 * np.dot(M_t, loga_der)
            res[sz_X:sz_X+sz_Y] = np.dot(D_Y, y) + sigma ** 2 * Omega_t
            # (beta_t)^T * C^T * L_X is a quadratic form evaluation
            vol_adj = np.dot(np.dot(beta_t, np.transpose(C_t)), loga_der)
            res[-1] = (kappa1 + kappa2 * sigma) * (theta - sigma) + sigma ** 2 * vol_adj

            return res

        sz_X = self.basis.nb_factors
        sz_Y = self.basis.nb_aux_factors
        ode_init = np.concatenate((x0, y0, np.array([self.sigma0])))
        # solve ODE with initial conditions
        oderesult = solve_ivp(fun=func_mean_rhs, t_span=(0, expiry), t_eval=t_grid, y0=ode_init,
                              args=(ts_sw, self.kappa1, self.kappa2, self.theta),
                              dense_output=False)
        mx = np.transpose(oderesult.y[:sz_X, :])
        my = np.transpose(oderesult.y[sz_X:sz_X+sz_Y, :])
        return mx, my


    def transform_QA_params(self,
                            expiry: float,
                            tenor: float,
                            t_grid: np.ndarray,
                            x0: np.ndarray = None,
                            y0: np.ndarray = None
                            ) -> Tuple[np.ndarray, ...]:
        """Calculate a, kappa_0, kappa_1, kappa_2, beta, volvol under Q^A for multi-factor model at times t_grid"""
        if x0 is None:
            x0 = np.zeros((self.basis.get_nb_factors()))
        else:
            assert x0.shape == (self.basis.get_nb_factors(),)
        if y0 is None:
            y0 = np.zeros((self.basis.get_nb_aux_factors()))
        else:
            assert y0.shape == (self.basis.get_nb_aux_factors(),)

        theta = self.theta
        kappa1 = self.kappa1
        kappa2 = self.kappa2
        self.q = self.theta
        q = self.q

        # interpolate a, beta, vol-vol from coarse grid to finer one
        ts_sw = get_default_swap_term_structure(expiry=expiry, tenor=tenor)
        if expiry not in t_grid:
            raise ValueError(f"expiry must be in grid")
        idx_ttm = np.where(t_grid == expiry)[0][0]
        t_grid = t_grid[:idx_ttm + 1]

        mx_grid, my_grid = self.calc_QA_mean_states(expiry=expiry, tenor=tenor, t_grid=t_grid, x0=x0, y0=y0)
        # mx_grid, my_grid = np.zeros((t_grid.size, self.basis.get_nb_factors())), np.zeros((t_grid.size, self.basis.get_nb_aux_factors()))
        swap_grad = np.ones((t_grid.size, self.basis.nb_factors)) * np.nan
        ann = np.ones_like(t_grid) * np.nan
        loga_der = np.ones((t_grid.size, self.basis.nb_factors)) * np.nan
        for idx, (t, mx, my) in enumerate(zip(t_grid, mx_grid, my_grid)):
            swap_grad[idx, :] = self.basis.swap_rate(t, ts_sw, mx, my, ccy=self.ccy)[1]
            ann[idx] = self.basis.annuity(t, ts_sw, mx, my, m=0, ccy=self.ccy)
            ann_der1 = self.basis.annuity(t, ts_sw, mx, my, m=1, ccy=self.ccy)
            loga_der[idx, :] = ann_der1 / ann[idx]

        # interpolate a, beta, vol-vol from coarse grid to finer one
        a_interp = np.ones((t_grid.size, self.basis.nb_factors)) * np.nan
        beta_interp = np.ones((t_grid.size, self.basis.nb_factors)) * np.nan
        volvol_interp = np.ones_like(t_grid) * np.nan
        beta2_interp = np.ones_like(t_grid) * np.nan
        for idx, t in enumerate(t_grid):
            idx_t = bracket(self.ts[1:], t, throw_if_not_found=True)
            beta_interp[idx, :] = self.beta.xs[idx_t]
            volvol_interp[idx] = self.volvol.xs[idx_t]
            a_interp[idx, :] = np.dot(swap_grad[idx, :], self.C[idx_t])
            beta2_interp[idx] = np.dot(beta_interp[idx, :], np.dot(np.transpose(self.C[idx_t]), loga_der[idx, :]))

        term0 = beta2_interp * theta * theta
        term1 = kappa1 - kappa2 * theta + 2.0 * (kappa2-beta2_interp)*theta
        term2 = kappa2 - beta2_interp

        return a_interp, term0, term1, term2, beta_interp, volvol_interp, ts_sw

    #############################################################################
    #                     Analytics under terminal measure Q^T for mid-curves
    #############################################################################
    def transform_QT_params(self,
                            expiry: float,
                            t_start: float,
                            t_end: float,
                            t_grid: np.ndarray) -> Tuple[np.ndarray, ...]:
        # t_start, t_end = get_futures_fixing_and_pmt(expiry=expiry, lag=lag)
        assert expiry <= t_start < t_end
        theta = self.theta
        kappa1 = self.kappa1
        kappa2 = self.kappa2
        # self.q = self.theta
        q = self.theta if self.q is None else self.q

        if expiry not in t_grid:
            raise ValueError(f"expiry must be in grid")
        idx_ttm = np.where(t_grid == expiry)[0][0]
        t_grid = t_grid[:idx_ttm + 1]

        a_interp = np.ones((t_grid.size, self.basis.nb_factors)) * np.nan
        beta_interp = np.ones((t_grid.size, self.basis.nb_factors)) * np.nan
        eta_interp = np.ones((t_grid.size, self.basis.nb_factors)) * np.nan
        volvol_interp = np.ones_like(t_grid) * np.nan
        delta_interp = np.ones_like(t_grid) * np.nan

        term0 = np.ones_like(t_grid) * np.nan
        term1 = np.ones_like(t_grid) * np.nan
        term2 = np.ones_like(t_grid) * np.nan

        # interpolate a, beta, vol-vol from coarse grid to finer one
        for idx, t in enumerate(t_grid):
            idx_t = bracket(self.ts[1:], t, throw_if_not_found=True)
            beta_interp[idx, :] = self.beta.xs[idx_t]
            volvol_interp[idx] = self.volvol.xs[idx_t]
            tau_end = t_end - t
            tau_start = t_start - t
            tau_exp = expiry - t
            B_P_end = self.basis.bond_coeffs(tau_end)[0]
            B_P_start = self.basis.bond_coeffs(tau_start)[0]
            B_P_exp = self.basis.bond_coeffs(tau_exp)[0]
            a_interp[idx, :] = np.dot(np.transpose(self.C[idx_t]), B_P_end-B_P_start)
            eta_interp[idx, :] = np.dot(np.transpose(self.C[idx_t]), B_P_exp)

            beta_x_eta = np.dot(beta_interp[idx, :], eta_interp[idx, :])

            term0[idx] = -beta_x_eta * q**2
            term1[idx] = kappa1 - kappa2 * q + 2.0*(kappa2 + beta_x_eta)
            term2[idx] = kappa2 + beta_x_eta

        return a_interp, eta_interp, term0, term1, term2, beta_interp, volvol_interp

    def check_QT_kappa2(self, t_start: float, t_end: float = None) -> bool:
        if t_end is None:
            t_end = t_start + 0.25
        expiry = t_start
        t_grid = generate_ttms_grid(np.array([expiry]))
        a_interp, delta_interp, term0, term1, term2, beta_interp, volvol_interp = self.transform_QT_params(expiry=expiry, t_start=t_start, t_end=t_end, t_grid=t_grid)
        return np.all(term2 > 0.0)

    def check_QA_kappa2(self, expiry: float, tenor: float) -> bool:
        t_grid = generate_ttms_grid(np.array([expiry]))

        a_interp, term0, term1, term2, beta_interp, volvol_interp, _ = self.transform_QA_params(expiry=expiry,
                                                                                                tenor=tenor,
                                                                                                t_grid=t_grid)
        return np.all(term2 > 0.0)


    def reduce(self, ids: List[str]):
        ttms = [MultiFactRateLogSvParams.get_frac(id) for id in ids]
        assert set(ttms) <= set(self.ts)
        indices = np.in1d(self.ts, ttms).nonzero()[0] - 1
        ts_indices = np.concatenate(([0], indices+1))
        assert np.all(indices >= 0)

        param = MultiFactRateLogSvParams(sigma0=self.sigma0,
                                         theta=self.theta,
                                         kappa1=self.kappa1,
                                         kappa2=self.kappa2,
                                         beta=TermStructure(self.beta.ts[ts_indices], self.beta.xs[indices]),
                                         volvol=TermStructure(self.volvol.ts[ts_indices], self.volvol.xs[indices]),
                                         A=self.A[indices, :],
                                         R=self.R,
                                         basis=self.basis,
                                         ccy=self.ccy,
                                         vol_interpolation=self.vol_interpolation,
                                         q=self.q)
        return param

    def update_params(self, idx: int, A_idx: np.ndarray = None, beta_idx: np.ndarray = None, volvol_idx: float = None,
                      kappa1: float = None, kappa2: float = None,
                      sigma0: float = None):
        nb_factors = self.basis.get_nb_factors()
        if A_idx is not None:
            assert A_idx.shape == (nb_factors,)
            self.A[idx, :] = A_idx
        if beta_idx is not None:
            assert beta_idx.shape == (nb_factors,)
            self.beta.xs[idx, :] = beta_idx
        if volvol_idx is not None:
            self.volvol.xs[idx] = volvol_idx
        if kappa1 is not None:
            self.kappa1 = kappa1
        if kappa2 is not None:
            self.kappa2 = kappa2
        if sigma0 is not None:
            self.sigma0 = sigma0


        self.__post_init__()

    @classmethod
    def get_frac(cls, id: str) -> float:
        if id == '3m':
            return 0.25
        elif id == '6m':
            return 0.5
        elif id == '1y':
            return 1.0
        elif id == '2y':
            return 2.0
        elif id == '3y':
            return 3.0
        elif id == '4y':
            return 4.0
        elif id == '5y':
            return 5.0
        elif id == '7y':
            return 7.0
        elif id == '10y':
            return 10.0
        elif id == '31d':
            return 31.0/365.0
        elif id == '40d':
            return 40.0/365.0
        elif id == '66d':
            return 66.0/365
        elif id == '75d':
            return 75.0/365
        elif id == '84d':
            return 84.0/365
        elif id == '87d':
            return 87.0/365
        elif id == '103d':
            return 103.0/365
        elif id == '156d':
            return 156.0/365
        elif id == '194d':
            return 194.0/365
        else:
            raise NotImplementedError(f"id not found")


    def calc_factor_vols(self, yield_vols: np.ndarray) -> np.ndarray:
        assert yield_vols.ndim == 1 and yield_vols.shape[0] == self.basis.get_nb_factors()
        B = self.basis.get_matrix_B()
        R_chol = np.linalg.cholesky(self.R)
        factor_vol = np.dot(np.dot(np.linalg.inv(B), np.diag(yield_vols)), R_chol)
        # factor_vol = np.dot(np.diag(yield_vols), R_chol)
        return factor_vol

    def calc_factor_vols_dln(self, yield_vols: np.ndarray,
                             yields: np.ndarray,
                             b_dln: np.ndarray,
                             nb_path: int):

        nb_factors = self.basis.get_nb_factors()
        assert yield_vols.ndim == 1 and yield_vols.shape[0] == nb_factors
        assert b_dln.shape == yield_vols.shape
        assert yields.shape == (nb_path, nb_factors)
        B = self.basis.get_matrix_B()
        R_chol = np.linalg.cholesky(self.R)
        inv_B = np.linalg.inv(B)
        A_DLN = np.zeros((nb_path, nb_factors, nb_factors))
        factor_vol = np.zeros((nb_path, nb_factors, nb_factors))
        b_times_y = np.multiply(yields, b_dln)
        for i, b_times_y_ith in enumerate(b_times_y):
            A_DLN[i, :, :] = np.diag(yield_vols + b_times_y_ith)
            factor_vol[i, :, :] = np.dot(np.dot(inv_B, A_DLN[i, :, :]), R_chol)

        return factor_vol

