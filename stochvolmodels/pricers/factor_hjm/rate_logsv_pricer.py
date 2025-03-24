import numpy as np
from numba.typed import List
from typing import Tuple, Optional, Union
from enum import Enum
from scipy.integrate import solve_ivp

from stochvolmodels.pricers.factor_hjm.rate_factor_basis import NelsonSiegel
from stochvolmodels.pricers.factor_hjm.rate_core import bracket, divide_mc, prod_mc, get_futures_start_and_pmt
from stochvolmodels.pricers.factor_hjm.rate_evaluate import swap_rate, annuity, bond
from stochvolmodels.pricers.factor_hjm.rate_affine_expansion import compute_logsv_a_mgf_grid, UnderlyingType
from stochvolmodels.pricers.factor_hjm.rate_logsv_params import RateLogSvParams, pw_const, get_default_swap_term_structure, MultiFactRateLogSvParams
from stochvolmodels.pricers.analytic.bachelier import infer_normal_implied_vol, infer_normal_ivols_from_slice_prices
from stochvolmodels.pricers.logsv.affine_expansion import ExpansionOrder, get_expansion_n
from stochvolmodels.pricers.model_pricer import ModelPricer, ModelParams
from stochvolmodels.data.option_chain import OptionChain, SwOptionChain, FutOptionChain
from stochvolmodels.utils.funcs import to_flat_np_array, set_time_grid, timer


from stochvolmodels.pricers.factor_hjm.double_exp_pricer import de_pricer

class Measure(Enum):
    RISK_NEUTRAL = 1
    ANNUITY = 2
    FORWARD = 3


class FutSettleType(Enum):
    EURODOLLAR = 1
    SOFR = 2


def conv_adj_rhs_MF(tau: float,
                    state0: np.ndarray,
                    ttm: float,
                    params: MultiFactRateLogSvParams,
                    Delta: float,
                    settlement_type: FutSettleType,
                    expansion_order: ExpansionOrder
                    ) -> np.ndarray:
    M = params.M
    C = params.C
    Omega = params.Omega
    sigma0 = params.sigma0
    q = params.theta if params.q is None else params.q
    ts = params.beta.ts
    betaxs = params.beta.xs
    volvolxs = params.volvol.xs

    sz_X = params.basis.nb_factors
    sz_Y = params.basis.nb_aux_factors
    D_X = params.basis.get_generating_matrix()
    D_Y = params.basis.get_aux_generating_matrix()

    if expansion_order == ExpansionOrder.FIRST:
        sz_vol = 3
        if settlement_type == FutSettleType.SOFR:
            raise NotImplementedError
    elif expansion_order == ExpansionOrder.ZERO:
        sz_vol = 2
    else:
        raise NotImplementedError

    sz = sz_X + sz_Y + sz_vol
    assert state0.size == sz
    B1, B2 = state0[:sz_X], state0[sz_X:sz_X + sz_Y]
    if expansion_order == ExpansionOrder.FIRST:
        h1, h2, h0 = state0[-3], state0[-2], state0[-1]
    elif expansion_order == ExpansionOrder.ZERO:
        h1, h0 = state0[-2], state0[-1]
        h2 = 0.0
    else:
        raise NotImplementedError

    v0 = sigma0 - q

    idx_t = bracket(ts[1:], ttm - tau, True)
    kappa0 = params.kappa1 * (params.theta - q) + params.kappa2 * q * (params.theta - q)
    kappa1 = params.kappa1 - params.kappa2 * params.theta + 2.0*params.kappa2 * q
    kappa2 = params.kappa2
    beta_t = betaxs[idx_t]
    volvol_t = volvolxs[idx_t]
    vartheta_sq = np.linalg.norm(beta_t) ** 2 + volvol_t ** 2
    C_t = C[idx_t]
    M_t = M[idx_t]
    Omega_t = Omega[idx_t]

    B0 = params.basis.get_basis(0.0)
    B0_ext = params.basis.get_aux_basis(0.0)

    # interim variables
    B1_M_B1 = np.dot(np.dot(np.transpose(B1), M_t), B1)
    B1_C_beta = np.dot(np.dot(B1, C_t), beta_t)
    B2_Omega = np.dot(B2, Omega_t)

    rhs = np.zeros_like(state0)
    rhs[:sz_X] = np.dot(B1, D_X)
    rhs[sz_X:sz_X + sz_Y] = np.dot(B2, D_Y)
    if settlement_type == FutSettleType.SOFR:
        rhs[:sz_X] += B0 if tau < Delta else 0.0
        rhs[sz_X:sz_X + sz_Y] += B0_ext if tau < Delta else 0.0

    if expansion_order == ExpansionOrder.FIRST:
        # h1
        rhs[-3] = 2.0 * q * (0.5 * B1_M_B1 + B2_Omega) + 2.0 * kappa0 * h2 - kappa1 * h1 + vartheta_sq * (
                q * h1 * h1 + 2.0 * q * h2 + 2.0 * q * q * h1 * h2) + 2.0 * q * (h1 + q * h2) * B1_C_beta
        # h2
        rhs[-2] = (0.5 * B1_M_B1 + B2_Omega) - 2.0 * kappa1 * h2 - kappa2 * h1 + vartheta_sq * (
                0.5 * h1 * h1 + h2 + 4.0 * q * h1 * h2 + 2.0 * q * q * h2 * h2) + (h1 + 4.0 * q * h2) * B1_C_beta
        # h0
        rhs[-1] = q * q * (0.5 * B1_M_B1 + B2_Omega) + kappa0 * h1 + vartheta_sq * q * q * (
                0.5 * h1 * h1 + h2) + q * q * h1 * B1_C_beta
    elif expansion_order == ExpansionOrder.ZERO:
        # h1
        rhs[-2] = 2.0*q * (0.5*B1_M_B1 + B2_Omega + h1*B1_C_beta + 0.5*vartheta_sq*h1*h1) - kappa1 * h1
        # h0
        rhs[-1] = q * q * (0.5*B1_M_B1 + B2_Omega + h1*B1_C_beta + 0.5*vartheta_sq*h1*h1) + kappa0 * h1


    return rhs


def futures_conv_adj(t_start: float,
                     basis_type: str,
                     params: MultiFactRateLogSvParams,
                     t0: float,
                     Delta: float,
                     settlement_type: FutSettleType,
                     expansion_order: ExpansionOrder,
                     dense_output: bool = False,
                     t_grid: np.ndarray = None) -> Tuple[np.ndarray, ...]:
    assert basis_type == "NELSON-SIEGEL"

    bond_coeffs = params.basis.bond_coeffs(Delta)
    if expansion_order == ExpansionOrder.FIRST:
        vol_init = np.zeros((3,))
    elif expansion_order == ExpansionOrder.ZERO:
        vol_init = np.zeros((2,))
    else:
        raise NotImplementedError

    if settlement_type == FutSettleType.EURODOLLAR:
        cond_init = np.concatenate((bond_coeffs[0],
                                    bond_coeffs[1],
                                    vol_init))
    elif settlement_type == FutSettleType.SOFR:
        cond_init = np.concatenate((np.zeros_like(bond_coeffs[0]),
                                    np.zeros_like(bond_coeffs[1]),
                                    vol_init))
    else:
        raise NotImplementedError

    assert t0 <= t_start  # check for SOFR options
    tau_S = t_start - t0
    tau_E = tau_S + Delta
    t_eval = np.maximum(t_start-t0, 1e-4) if settlement_type == FutSettleType.EURODOLLAR else np.maximum(t_start+Delta-t0, 1e-4)

    if t_grid is not None:
        idx_ttm = np.where(t_grid == t_eval)[0][0]
        t_grid = t_grid[:idx_ttm + 1]

    # t_grid = np.arange(0.0, t_eval+1e-6, 0.025)

    sol_futconvex = solve_ivp(fun=conv_adj_rhs_MF, y0=cond_init,
                              args=(t_start, params, Delta, settlement_type, expansion_order),
                              t_span=(0.0, t_eval), dense_output=dense_output,
                              t_eval=t_grid, max_step=0.001)
    sol = sol_futconvex.y[:, -1]
    sz_X = params.basis.nb_factors
    sz_Y = params.basis.nb_aux_factors

    if expansion_order == ExpansionOrder.FIRST:
        b1, b2, h1, h2, h0 = sol[:sz_X], sol[sz_X:sz_X + sz_Y], sol[-3], sol[-2], sol[-1]
    elif expansion_order == ExpansionOrder.ZERO:
        b1, b2, h1, h0 = sol[:sz_X], sol[sz_X:sz_X + sz_Y], sol[-2], sol[-1]
        h2 = 0.0
    # now we take into account exp[-(...)X_t-(...)Y_t]
    # need to convert from tuple to list to support arithmetic operations
    b1 = b1 - (params.basis.bond_coeffs(tau_E)[0] - params.basis.bond_coeffs(tau_S)[0])
    b2 = b2 - (params.basis.bond_coeffs(tau_E)[1] - params.basis.bond_coeffs(tau_S)[1])

    if dense_output:
        assert t_grid is not None
        # we solve ODE in time-to-maturity tau=T-t, but need to return it in time t
        sol = sol_futconvex.sol(t_start-t_grid)
        b1 = sol.T[:, :sz_X]
        b2 = sol.T[:, sz_X:sz_X + sz_Y]
        h1 = sol.T[:, -2]
        h0 = sol.T[:, -1]
        if expansion_order == ExpansionOrder.FIRST:
            h2 = sol.T[:, -3]
        elif expansion_order == ExpansionOrder.ZERO:
            h2 = np.zeros_like(t_grid)

    return b1, b2, h1, h2, h0


def calc_futures_rate(ccy: str,
                      basis_type: str,
                      params: MultiFactRateLogSvParams,
                      x0: np.ndarray,
                      y0: np.ndarray,
                      sigma0: np.ndarray,
                      t0: float,
                      t_start: float,
                      t_end: float,
                      Delta: float,
                      settlement_type: FutSettleType,
                      expansion_order: ExpansionOrder) -> Tuple[np.ndarray, ...]:
    assert basis_type == "NELSON-SIEGEL"
    assert 0 <= t0 <= t_start
    q = params.theta if params.q is None else params.q
    v0 = sigma0[:, 0] - q
    b1, b2, h1, h2, h0 = futures_conv_adj(t_start=t_start,
                                          basis_type=basis_type,
                                          params=params,
                                          t0=t0,
                                          Delta=Delta,
                                          settlement_type=settlement_type,
                                          expansion_order=expansion_order)
    c_tau = np.exp(np.dot(b1, np.transpose(x0)) + np.dot(b2, np.transpose(y0)) + h0 + h1 * v0 + h2 * v0 * v0)
    P_t_Ts_Te = params.basis.bond(t=t0, T=t_end, x=x0, y=y0, ccy=ccy, m=0) / params.basis.bond(t=t0, T=t_start, x=x0,
                                                                                               y=y0, ccy=ccy, m=0)
    if x0 is None:
        x0 = np.zeros((params.basis.get_nb_factors(),))
    else:
        assert x0.ndim == 1 and x0.shape[0] == params.basis.get_nb_factors()
    if y0 is None:
        y0 = np.zeros((params.basis.get_nb_aux_factors(),))
    else:
        assert y0.ndim == 1 and y0.shape[0] == params.basis.get_nb_aux_factors()
    P_0_Ts_Te = params.basis.bond(t=t0, T=t_end, x=x0, y=y0, ccy=ccy, m=0)[0] / params.basis.bond(t=t0, T=t_start, x=x0, y=y0, ccy=ccy, m=0)[0]


    # futures convexity adjustment esimates E[exp(B_P*(X_T-X_t)+...)] thus we multiply by P(t,T_S,T_E)
    futures_analyt_ae1 = 1.0 / Delta * (1.0 / P_t_Ts_Te * c_tau - 1.0)

    return futures_analyt_ae1, c_tau, P_t_Ts_Te, P_0_Ts_Te


def logsv_chain_de_pricer(params: MultiFactRateLogSvParams,
                          t_grid: np.ndarray,
                          ttms: np.ndarray,
                          forwards: List[np.ndarray],
                          strikes_ttms: List[List[np.ndarray]],
                          optiontypes_ttms: List[np.ndarray],
                          is_stiff_solver: bool = False,
                          underlying_type: UnderlyingType = UnderlyingType.SWAP,
                          is_analytic: bool = False,
                          is_spot_measure: bool = True,
                          expansion_order: ExpansionOrder = ExpansionOrder.SECOND,
                          vol_scaler: float = None,
                          do_control_variate=False,
                          x0: np.ndarray = None,
                          y0: np.ndarray = None,
                          **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    model_prices_tenors_ttms = List()
    model_ivs_tenors_ttms = List()
    t_grid0 = t_grid
    if 'lag' in kwargs:
        lag = kwargs['lag']
    else:
        lag = 0.0
    if 'settlement_type' in kwargs:
        settlement_type = kwargs['settlement_type']
    else:
        settlement_type = FutSettleType.EURODOLLAR
    if underlying_type == UnderlyingType.SWAP:
        # sanity check: we must have as much forwards as tenors for swaptions
        assert params.basis.key_terms.size == len(forwards)
        # we calibrate expiry by expiry
        # per expiry we calculate implied vols tenor-by-tenor
        assert ttms.size == 1 and len(optiontypes_ttms) == 1
        ttms_ = np.ones_like(params.basis.key_terms) * ttms[0]
        optiontypes_ttms_ = [optiontypes_ttms[0] for tenor in params.basis.key_terms]
        rng_ttm = params.basis.key_terms

    elif underlying_type == UnderlyingType.FUTURES:
        assert len(forwards) == 1
        # we calibrate expiry by expiry
        assert ttms.size == 1 and len(optiontypes_ttms) == 1
        ttms_ = ttms
        optiontypes_ttms_ = optiontypes_ttms
        rng_ttm = ['FUTURES_DUMMY_TENOR']
    else:
        raise NotImplementedError

    for idx_tenor, _ in enumerate(rng_ttm):
        model_prices_ttms = List()
        model_ivs_ttms = List()

        tenor = np.nan
        if underlying_type == UnderlyingType.SWAP:
            tenor = rng_ttm[idx_tenor]

        for ttm, forward, strikes_ttm, optiontypes_ttm in zip(ttms_, forwards[idx_tenor], strikes_ttms[idx_tenor],
                                                              optiontypes_ttms_):
            if underlying_type == UnderlyingType.SWAP:
                a, kappa0, kappa1, kappa2, beta, volvol, _ = params.transform_QA_params(expiry=ttm, t_grid=t_grid0,
                                                                                        tenor=tenor, x0=x0, y0=y0)
                a0 = a
                a1 = np.zeros_like(kappa0)
                b = np.zeros_like(kappa0)

            elif underlying_type == UnderlyingType.FUTURES:
                tenor = 0.25
                start, end = get_futures_start_and_pmt(t0=ttm, lag=0.0, libor_tenor=tenor)
                frac = end - start
                a, eta, kappa0, kappa1, kappa2, beta, volvol = params.transform_QT_params(expiry=ttm, t_grid=t_grid,
                                                                                          t_start=start, t_end=end)

                b1, b2, h1, h2, h0 = futures_conv_adj(t_start=start,
                                                      basis_type="NELSON-SIEGEL",  # hard-coded
                                                      params=params,
                                                      t0=0.0,
                                                      Delta=tenor, expansion_order=ExpansionOrder.ZERO,
                                                      dense_output=True,
                                                      t_grid=t_grid, settlement_type=settlement_type)
                # for futures we have extra contribution
                a0 = a + np.einsum('i,ij->ij', h1, beta)
                a1 = np.multiply(h1, volvol)
                b = np.einsum('ij,ij->i', a0, eta) + 0.5 * np.einsum('ij,ij->i', a0, a0)
            else:
                NotImplementedError(f"Underlying type is not supported")
            # cut t_grid
            itemindex = np.where(t_grid0 == ttm)
            itemindex = itemindex[0][0]
            t_grid = t_grid0[:itemindex + 1]

            # delta = np.zeros_like(delta)  # TODO: remove. just for testing
            # params.q = 0.0  # TODO: q=0 just for testing. reset it for pricing
            def ff(p: np.ndarray) -> np.ndarray:
                real_phi = -0.5
                phi_grid = real_phi + 1j * p
                log_mgf_grid = compute_logsv_a_mgf_grid(ttm=ttm,
                                                        phi_grid=phi_grid,
                                                        sigma0=params.sigma0,
                                                        q=params.q,
                                                        times=t_grid,
                                                        a0=a0,
                                                        a1=a1,
                                                        kappa0=kappa0,
                                                        kappa1=kappa1,
                                                        kappa2=kappa2,
                                                        beta=beta,
                                                        volvol=volvol,
                                                        b=b,
                                                        underlying_type=underlying_type,
                                                        expansion_order=expansion_order,
                                                        is_stiff_solver=is_stiff_solver)[1]
                integrand_ = np.zeros((phi_grid.shape[0], strikes_ttm.shape[0]))
                if underlying_type == UnderlyingType.SWAP:
                    moneyness = strikes_ttm - forward
                    p_payoff = (1.0 / np.pi) / (phi_grid * phi_grid)
                    for idx, (x, strike, type_) in enumerate(zip(moneyness, strikes_ttm, optiontypes_ttm)):
                        integrand_[:, idx] = np.real(p_payoff * (np.exp(x*phi_grid + log_mgf_grid)))
                elif underlying_type == UnderlyingType.FUTURES:
                    moneyness = np.log((strikes_ttm + 1.0/frac)/(forward + 1.0/frac))
                    p_payoff = (1.0 / np.pi) / (phi_grid * (phi_grid + 1.0))
                    for idx, (x, strike, type_) in enumerate(zip(moneyness, strikes_ttm, list(optiontypes_ttm))):
                        integrand_[:, idx] = np.real(
                            p_payoff * (-(strike + 1.0 / frac)) * (np.exp(x * phi_grid + log_mgf_grid)))

                else:
                    NotImplementedError("Underlying type is not supported")



                np.set_printoptions(linewidth=np.inf, precision=16)
                return integrand_

            # in case of options on swap rate, we directly price call options
            # in case of options on futures, we price capped payoff and call option through capped payoff
            if underlying_type == UnderlyingType.SWAP:
                def ff_transf(model_prices: np.ndarray):
                    normal_ivols = infer_normal_ivols_from_slice_prices(ttm=ttm, forward=forward,
                                                                        strikes=strikes_ttm,
                                                                        model_prices=model_prices[0,:],
                                                                        optiontypes=np.repeat('C', strikes_ttm.size),
                                                                        discfactor=1.0)
                    return model_prices, normal_ivols
            elif underlying_type == UnderlyingType.FUTURES:
                def ff_transf(capped_prices: np.ndarray) -> (np.ndarray, np.ndarray):
                    call_prices = forward + 1.0/frac - capped_prices
                    normal_ivols = infer_normal_ivols_from_slice_prices(ttm=ttm, forward=forward, strikes=strikes_ttm,
                                                                        model_prices=call_prices[0, :],
                                                                        optiontypes=np.repeat('C', strikes_ttm.size),
                                                                        discfactor=1.0)
                    return call_prices, normal_ivols
            else:
                NotImplementedError(f"underlying type is unknown")


            model_prices_ttm, model_ivs_ttm = de_pricer(ff, ff_transf)
            model_prices_ttms.append(model_prices_ttm[0, :])
            model_ivs_ttms.append(model_ivs_ttm)
        model_prices_tenors_ttms.append(model_prices_ttms)
        model_ivs_tenors_ttms.append(model_ivs_ttms)
    return model_prices_tenors_ttms, model_ivs_tenors_ttms


class RateLogSVPricer(ModelPricer):
    def price_chain(self,
                    option_chain: SwOptionChain,
                    params: Union[RateLogSvParams, MultiFactRateLogSvParams],
                    is_spot_measure: bool = True,
                    **kwargs
                    ) -> List[np.ndarray]:
        """
        implementation of generic method price_chain using log sv wrapper
        """
        t_grid = kwargs['t_grid']
        idxs = kwargs['idxs']
        ttms = np.array(option_chain.ttms[idxs])
        forwards = [option_chain.forwards[idx_tenor][idxs] for idx_tenor, _ in enumerate(option_chain.tenors)]
        strikes_ttms = [option_chain.strikes_ttms[idx_tenor][idxs] for idx_tenor, _ in enumerate(option_chain.tenors)]
        optiontypes_ttms = option_chain.optiontypes_ttms[idxs]

        model_ivols = logsv_chain_de_pricer(params=params,
                                            t_grid=t_grid,
                                            ttms=ttms,
                                            forwards=forwards,
                                            strikes_ttms=strikes_ttms,
                                            optiontypes_ttms=optiontypes_ttms,
                                            expansion_order=ExpansionOrder.FIRST,
                                            is_stiff_solver=False)[1]
        return model_ivols

    def model_mc_price_chain(self,
                             option_chain: SwOptionChain,
                             params: RateLogSvParams,
                             nb_path: int = 100000,
                             **kwargs
                             ) -> (List[np.ndarray], List[np.ndarray]):
        return logsv_mc_chain_pricer(sigma0=params.sigma0,
                                     theta=params.theta,
                                     kappa1=params.kappa1,
                                     kappa2=params.kappa2,
                                     ts=params.alpha.ts,
                                     betaxs=params.beta.xs,
                                     volvolxs=params.volvol.xs,
                                     ttms=option_chain.ttms,
                                     forwards=option_chain.forwards,
                                     strikes_ttms=option_chain.strikes_ttms,
                                     optiontypes_ttms=option_chain.optiontypes_ttms,
                                     nb_path=nb_path,
                                     **kwargs)


class RateFutLogSVPricer(ModelPricer):
    def price_chain(self,
                    option_chain: FutOptionChain,
                    params: MultiFactRateLogSvParams,
                    is_spot_measure: bool = True,
                    **kwargs
                    ) -> List[np.ndarray]:
        """
        implementation of generic method price_chain using log sv wrapper
        """
        t_grid = kwargs['t_grid']
        idxs = kwargs['idxs']
        ttms = np.array(option_chain.ttms[idxs])
        forwards = [option_chain.forwards[idxs]]
        strikes_ttms = [option_chain.strikes_ttms[idxs]]
        optiontypes_ttms = [option_chain.optiontypes_ttms[0]]
        if 'expansion_order' in kwargs:
            expansion_order = kwargs['expansion_order']
        else:
            expansion_order = ExpansionOrder.FIRST
        if 'is_stiff_solver' in kwargs:
            is_stiff_solver = kwargs['is_stiff_solver']
        else:
            is_stiff_solver = True
        if 'x0' in kwargs:
            x0 = kwargs['x0']
        else:
            x0 = None
        if 'y0' in kwargs:
            y0 = kwargs['y0']
        else:
            y0 = None


        model_ivols = logsv_chain_de_pricer(params=params,
                                            t_grid=t_grid,
                                            ttms=ttms,
                                            forwards=forwards,
                                            strikes_ttms=strikes_ttms,
                                            optiontypes_ttms=optiontypes_ttms,
                                            underlying_type=UnderlyingType.FUTURES,
                                            lag=0, # TODO: review it for mid-curve options
                                            expansion_order=expansion_order,
                                            is_stiff_solver=is_stiff_solver,
                                            x0=x0,
                                            y0=y0)[1]
        return model_ivols

    @classmethod
    def populate_betas(cls, beta: float, basis: NelsonSiegel) -> np.ndarray:
        if basis.get_nb_factors() == 3:
            return np.array([beta, -0.5*beta, 0.0])
        elif basis.get_nb_factors() == 1:
            return np.array([beta])
        else:
            raise NotImplementedError


def update_params(param0: MultiFactRateLogSvParams,
                  idx: int,
                  opt_val: np.ndarray):
    nb_factors = param0.basis.get_nb_factors()
    a_ttm, beta_ttm, volvol_ttm = opt_val[:nb_factors] * 0.01, opt_val[nb_factors:nb_factors + nb_factors], opt_val[-1]
    # update parameters
    param0.update_params(idx=idx, A_idx=a_ttm, beta_idx=beta_ttm, volvol_idx=volvol_ttm)


def logsv_mc_chain_pricer(sigma0: float,
                          theta: float,
                          kappa1: float,
                          kappa2: float,
                          ts: np.ndarray,
                          axs: np.ndarray,
                          bxs: np.ndarray,
                          betaxs: np.ndarray,
                          volvolxs: np.ndarray,
                          ttms: np.ndarray,
                          forwards: np.ndarray,
                          strikes_ttms: List[np.ndarray],
                          optiontypes_ttms: List[np.ndarray],
                          lamda: np.ndarray = None,
                          is_annuity_measure: bool = False,
                          is_approx_dynamics: bool = False,
                          nb_path: int = 100000,
                          seed: int = None,
                          ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # starting values
    sigma0 = sigma0 * np.ones(nb_path)

    # outputs as numpy lists
    option_prices_ttm = List()
    option_std_ttm = List()
    for ttm, forward, strikes_ttm, optiontypes_ttm in zip(ttms, forwards, strikes_ttms, optiontypes_ttms):
        x0 = np.zeros((nb_path,))
        y0 = np.zeros((nb_path,))
        I0 = np.zeros((nb_path,))
        if is_approx_dynamics:
            s0 = simulate_logsv_swap_approx_terminal(ttm=ttm,
                                                     sigma0=sigma0,
                                                     theta=theta,
                                                     kappa1=kappa1,
                                                     kappa2=kappa2,
                                                     ts=ts,
                                                     axs=axs,
                                                     betaxs=betaxs,
                                                     volvolxs=volvolxs,
                                                     seed=seed,
                                                     nb_path=nb_path)
            option_prices, option_std = compute_mcapprox_payoff(ttm=ttm,
                                                                s_mc=s0,
                                                                strikes_ttm=strikes_ttm,
                                                                optiontypes_ttm=optiontypes_ttm)
        else:
            x0, y0, I0, qvar0, _ = simulate_logsv_x_vol_terminal(ttm=ttm,
                                                                 x0=x0,
                                                                 y0=y0,
                                                                 I0=I0,
                                                                 sigma0=sigma0,
                                                                 theta=theta,
                                                                 kappa1=kappa1,
                                                                 kappa2=kappa2,
                                                                 ts=ts,
                                                                 axs=axs,
                                                                 bxs=bxs,
                                                                 betaxs=betaxs,
                                                                 volvolxs=volvolxs,
                                                                 lamda=lamda,
                                                                 nb_path=nb_path,
                                                                 seed=seed,
                                                                 is_annuity_measure=is_annuity_measure)

            # np.savetxt('c:/temp/dump.txt', qvar0)

            option_prices, option_std = compute_mc_vars_payoff(ttm=ttm,
                                                               x0=x0,
                                                               y0=y0,
                                                               I0=I0,
                                                               strikes_ttm=strikes_ttm,
                                                               optiontypes_ttm=optiontypes_ttm,
                                                               is_annuity_measure=is_annuity_measure)
        option_prices_ttm.append(option_prices)
        option_std_ttm.append(option_std)

    return option_prices_ttm, option_std_ttm


# @njit(cache=False, fastmath=True)  # TODO
def simulate_logsv_x_vol_terminal(ttm: float,
                                  x0: np.ndarray,
                                  y0: np.ndarray,
                                  I0: np.ndarray,
                                  sigma0: np.ndarray,
                                  theta: float,
                                  kappa1: float,
                                  kappa2: float,
                                  ts: np.ndarray,
                                  axs: np.ndarray,
                                  bxs: np.ndarray,
                                  betaxs: np.ndarray,
                                  volvolxs: np.ndarray,
                                  lamda: np.ndarray = None,
                                  is_heston: bool = False,
                                  is_annuity_measure: bool = False,
                                  nb_path: int = 100000,
                                  qvar0: np.ndarray = None,
                                  seed: int = None
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ttms = np.array([ttm])
    res = simulate_logsv_x_vol(ttms=ttms, x0=x0, y0=y0, I0=I0, sigma0=sigma0, theta=theta, kappa1=kappa1, kappa2=kappa2,
                               ts=ts, axs=axs, bxs=bxs, betaxs=betaxs, volvolxs=volvolxs, lamda=lamda,
                               is_heston=is_heston,
                               is_annuity_measure=is_annuity_measure,
                               nb_path=nb_path,
                               qvar0=qvar0,
                               seed=seed)
    x0 = res[0][-1]
    y0 = res[1][-1]
    I0 = res[2][-1]
    qvar0 = res[3][-1]
    sigma0 = res[4][-1]

    return x0, y0, I0, qvar0, sigma0


# @njit(cache=False, fastmath=True)  # TODO
def simulate_logsv_x_vol(ttms: np.ndarray,
                         x0: np.ndarray,
                         y0: np.ndarray,
                         I0: np.ndarray,
                         sigma0: np.ndarray,
                         theta: float,
                         kappa1: float,
                         kappa2: float,
                         ts: np.ndarray,
                         axs: np.ndarray,
                         bxs: np.ndarray,
                         betaxs: np.ndarray,
                         volvolxs: np.ndarray,
                         basis, # : Cheyette1D,
                         ts_sw: np.ndarray,
                         ccy: str,
                         is_heston: bool = False,
                         is_annuity_measure: bool = False,
                         nb_path: int = 100000,
                         seed: int = None,
                         cap_lsv: bool = False,
                         W: List[np.ndarray] = None,
                         ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    assert ttms.shape[0] > 0
    ttm = ttms[-1]
    if x0.shape[0] == 1:  # initial value
        x0 = x0 * np.zeros(nb_path)
    else:
        assert x0.shape[0] == nb_path

    if y0.shape[0] == 1:  # initial value
        y0 = y0 * np.zeros(nb_path)
    else:
        assert y0.shape[0] == nb_path

    if I0.shape[0] == 1:  # initial value
        I0 = I0 * np.zeros(nb_path)
    else:
        assert I0.shape[0] == nb_path

    if sigma0.shape[0] == 1:
        sigma0 = sigma0 * np.ones(nb_path)
    else:
        assert sigma0.shape[0] == nb_path

    # pre-generate random numbers
    if seed is None:
        seed = 16
    np.random.seed(seed)  # fix seed
    nb_steps, dt, grid_t = set_time_grid(ttm=ttm)
    if W is None:
        W0 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))  # TODO: undo
        W1 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))  # TODO: undo
    else:
        W0 = W[0] * np.sqrt(dt)
        W1 = W[1] * np.sqrt(dt)
    # W0 = np.transpose(np.loadtxt('C:/temp/rnd1.txt', delimiter=',')) * np.sqrt(dt)
    # W1 = np.transpose(np.loadtxt('C:/temp/rnd2.txt', delimiter=',')) * np.sqrt(dt)

    idx_ttms = [np.where(grid_t == t)[0][0] for t in ttms]
    # print(idx_ttms)
    x0s = List()
    y0s = List()
    I0s = List()
    qvar0s = List()
    sigma0s = List()

    mrv_r = basis.meanrev
    # ts_sw = get_default_swap_term_structure(ttm)  # need when we simulate under Q^A
    sigma0_cpy = sigma0

    if 0 in idx_ttms:
        x0s.append(x0), y0s.append(y0), I0s.append(I0), sigma0s.append(sigma0)

    if not is_heston:
        log_vol = np.log(sigma0)

        for idx, (t_, w0, w1) in enumerate(zip(grid_t, W0, W1)):
            # reshape (n,) array to (n,1)
            w0 = np.reshape(w0, (nb_path, 1))
            w1 = np.reshape(w1, (nb_path, 1))

            a_t = pw_const(ts, axs, t_, flat_extrapol=False, shift=1)
            b_t = pw_const(ts, bxs, t_, flat_extrapol=False, shift=1)
            beta_t = pw_const(ts, betaxs, t_, flat_extrapol=False, shift=1)
            volvol_t = pw_const(ts, volvolxs, t_, flat_extrapol=False, shift=1)
            vartheta2 = beta_t * beta_t + volvol_t * volvol_t

            if is_annuity_measure:
                ann_der0_mc = basis.annuity(t=t_, ts_sw=ts_sw, x=x0, y=y0, m=0, ccy=ccy)
                ann_der1_mc = basis.annuity(t=t_, ts_sw=ts_sw, x=x0, y=y0, m=1, ccy=ccy)
                d_loga_dx = divide_mc(ann_der1_mc, ann_der0_mc)
                adj_x_drift = (a_t + b_t * x0) ** 2 * sigma0 ** 2 * d_loga_dx
                adj_vol_drift = (a_t + b_t * x0) * beta_t * sigma0 * d_loga_dx
            else:
                adj_x_drift = 0
                adj_vol_drift = 0

            I0 = I0 + x0[:,0] * dt
            # qvar0 = qvar0 + sigma0 * sigma0 * dt
            lsv = (a_t + b_t * x0) * sigma0
            lsv = np.minimum(lsv, 5.0) if cap_lsv else lsv  # cap to 500%
            y0 = y0 + (lsv ** 2 - 2.0 * mrv_r * y0) * dt
            x0 = x0 + (y0 - mrv_r * x0) * dt + lsv * w0 + adj_x_drift * dt
            log_vol = log_vol + ((kappa1 * theta / sigma0) - (
                    kappa1 - kappa2 * theta + 0.5 * vartheta2) - kappa2 * sigma0) * dt + beta_t * w0 + volvol_t * w1 + adj_vol_drift * dt
            sigma0 = np.exp(log_vol)
            # check that forwards are ok
            tau = 5.0
            mrv = mrv_r
            exp_mmt = np.exp(-mrv * tau)
            G_t_T = (1.0 - exp_mmt) / mrv
            fwd = exp_mmt * x0 + exp_mmt * G_t_T * y0
            # end of check
            if idx + 1 in idx_ttms:
                x0s.append(x0), y0s.append(y0), I0s.append(I0), sigma0s.append(sigma0), None

    else:  # case of heston dynamics
        # raise ValueError("Heston not implemented properly")
        if np.fabs(kappa2) >= 1e-8:
            raise ValueError("in heston model kappa2 must be zero")
        var0 = sigma0
        for idx, (t_, w0, w1) in enumerate(zip(grid_t, W0, W1)):
            vol0 = np.sqrt(var0)
            a_t = pw_const(ts, axs, t_, False)
            b_t = pw_const(ts, bxs, t_, False)
            beta_t = pw_const(ts, betaxs, t_, False)
            volvol_t = pw_const(ts, volvolxs, t_, False)

            if is_annuity_measure:
                ann_der0_mc = annuity(t=t_, ts_sw=ts_sw, x=x0, y=y0, m=0, is_mc_mode=True)
                ann_der1_mc = annuity(t=t_, ts_sw=ts_sw, x=x0, y=y0, m=1, is_mc_mode=True)
                d_loga_dx = ann_der1_mc / ann_der0_mc
                adj_x_drift = (a_t + b_t * x0) ** 2 * var0 * d_loga_dx
                adj_vol_drift = (a_t + b_t * x0) * beta_t * var0 * d_loga_dx
            else:
                adj_x_drift = 0
                adj_vol_drift = 0

            I0 = I0 + x0 * dt
            lsv = (a_t + b_t * x0) * vol0
            # capping with some useful diagnostics
            if not np.all(np.isfinite(lsv)):
                print(f"step = {idx}: explosion in Heston")
            if cap_lsv:
                cap = 5.0  # cap to 500%
                if np.any(lsv >= cap):
                    print(f"t={t_:.2f}: {np.argwhere(lsv >= cap).size} paths reached the cap: " + ''.join(
                        str(x) for x in np.argwhere(lsv >= cap)))
                lsv = np.minimum(lsv, cap)

            y0 = y0 + (lsv ** 2 - 2.0 * mrv_r * y0) * dt
            x0 = x0 + (y0 - mrv_r * x0) * dt + lsv * w0 + adj_x_drift * dt
            var0 = var0 + kappa1 * (theta - var0) * dt + vol0 * (beta_t * w0 + volvol_t * w1) + adj_vol_drift * dt
            var0 = np.maximum(var0, 1e-4)

            if idx + 1 in idx_ttms:
                x0s.append(x0), y0s.append(y0), I0s.append(I0)
                qvar0s.append(np.ones_like(x0)*np.nan)
                sigma0s.append(np.ones_like(x0)*np.nan)

    return x0s, y0s, I0s, qvar0s, sigma0s

def make_mc_array(x: np.ndarray, nb_path: int):
    x_ = np.zeros((nb_path, x.size))
    for idx, val in enumerate(x):
        x_[:, idx] = val
    return x_


def simulate_logsv_MF(ttms: np.ndarray,
                      x0: np.ndarray,
                      y0: np.ndarray,
                      I0: np.ndarray,
                      sigma0: np.ndarray,
                      theta: float,
                      kappa1: float,
                      kappa2: float,
                      ts: np.ndarray,
                      A: List[np.ndarray],  # for multi-factor model
                      R: List[np.ndarray],  # for multi-factor model
                      C: List[np.ndarray],  # for multi-factor model
                      Omega: List[np.ndarray],  # for multi-factor model
                      betaxs: np.ndarray,
                      volvolxs: np.ndarray,
                      basis: NelsonSiegel,
                      ts_sw: np.ndarray,
                      T_fwd: float,
                      ccy: str,
                      measure_type: Measure = Measure.RISK_NEUTRAL,
                      nb_path: int = 100000,
                      seed: int = None,
                      W: List[np.ndarray] = None,
                      bxs: np.ndarray = None,
                      params0: MultiFactRateLogSvParams = None,
                      year_days: int = 360,
                      quasi_random: bool = False
                      ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Simulate factors X_t, Y_t, sigma_t, I_t for each t in ttms
    TODO: ideally we should not pass params0, but for DLN skew we need to recalculate volatility matrix C and
    TODO: auxiliary vector \\Omega_t for each simulation step, thus passed as optional parameter
    TODO: in potential numba implementation we should think of removing it
    """
    assert ttms.shape[0] > 0
    ttm = ttms[-1]

    assert ts.shape[0] > 0 and ts[0] == 0.0
    nb_times = ts.size - 1

    # check inputs
    nb_factors = basis.get_nb_factors()
    assert A.ndim == 2 and A.shape[0] == nb_times and A.shape[1] == nb_factors
    assert betaxs.ndim == 2 and betaxs.shape[0] == nb_times and betaxs.shape[1] == nb_factors
    assert volvolxs.ndim == 1 and volvolxs.shape[0] == nb_times
    assert R.ndim == 2 and R.shape[0] == R.shape[1] == nb_factors
    assert Omega.ndim == 2 and Omega.shape[0] == nb_times and Omega.shape[1] == basis.get_nb_aux_factors()
    # if b (displaced log-normal skew) is provided, we assume it is flat across expiries and values are per factor
    if bxs is not None:
        assert measure_type is Measure.RISK_NEUTRAL and np.all(np.fabs(betaxs) <= 1e-8) and np.all(volvolxs <= 1e-8)
        assert kappa1 <= 1e-8 and kappa2 <= 1e-8
        assert bxs.shape == (nb_factors,)

    if x0.shape[0] == basis.get_nb_factors():  # initial value
        x0 = make_mc_array(x0, nb_path)
    else:
        assert x0.shape == (nb_path, basis.get_nb_factors())

    if y0.shape[0] == basis.get_nb_aux_factors():  # initial value
        y0 = make_mc_array(y0, nb_path)
    else:
        assert y0.shape == (nb_path, basis.get_nb_aux_factors())

    if I0.shape[0] == 1:  # initial value
        I0 = I0 * np.zeros(nb_path)
    else:
        assert I0.shape[0] == nb_path

    if sigma0.shape[0] == 1:
        sigma0 = sigma0 * np.ones(nb_path)
    else:
        assert sigma0.shape[0] == nb_path

    # pre-generate random numbers
    if seed is None:
        seed = 16
    np.random.seed(seed)  # fix seed
    nb_steps, dt, grid_t = set_time_grid(ttm=ttm, nb_steps_per_year=360)
    if W is None:
        W0 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path, basis.get_nb_factors()))  # TODO: undo
        W1 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))  # TODO: undo

    else:
        W0 = W[0] * np.sqrt(dt)
        W1 = W[1] * np.sqrt(dt)
    # W0 = np.transpose(np.loadtxt('C:/temp/rnd1.txt', delimiter=',')) * np.sqrt(dt)
    # W1 = np.transpose(np.loadtxt('C:/temp/rnd2.txt', delimiter=',')) * np.sqrt(dt)

    idx_ttms = [np.where(np.isclose(grid_t, t))[0][0] for t in ttms]
    x0s = []
    y0s = []
    I0s = []
    sigma0s = []

    if 0 in idx_ttms:
        x0s.append(x0), y0s.append(y0), I0s.append(I0), sigma0s.append(sigma0)

    log_vol = np.log(sigma0)
    D_X = basis.get_generating_matrix()
    D_Y = basis.get_aux_generating_matrix()
    # for simulation under Q we need short rate r(t)=f(t,t) to calculate I(t)
    B0_X = basis.get_basis(0.0)
    B0_Y = basis.get_aux_basis(0.0)

    for idx, (t_, w0, w1) in enumerate(zip(grid_t, W0, W1)):
        w1 = np.reshape(w1, (nb_path, 1))
        # interpolation of the volatility matrix is tedious
        idx_t = bracket(ts[1:], t_, throw_if_not_found=True)
        beta_t = betaxs[idx_t]
        volvol_t = volvolxs[idx_t]
        C_t = C[idx_t]
        Omega_t = Omega[idx_t]
        vartheta2 = np.dot(beta_t, beta_t) + volvol_t * volvol_t

        if measure_type == Measure.ANNUITY:
            ann_der0_mc = basis.annuity(t=t_, ts_sw=ts_sw, x=x0, y=y0, ccy=ccy, m=0)
            ann_der1_mc = basis.annuity(t=t_, ts_sw=ts_sw, x=x0, y=y0, ccy=ccy, m=1)
            d_loga_dx = divide_mc(ann_der1_mc, ann_der0_mc)
            adj_x_drift = prod_mc(np.dot(d_loga_dx, np.dot(C_t, np.transpose(C_t))), sigma0[:, 0] ** 2)
            adj_vol_drift = prod_mc(sigma0, np.dot(d_loga_dx, np.dot(C_t, beta_t)))
        elif measure_type == Measure.RISK_NEUTRAL:
            adj_x_drift = 0
            adj_vol_drift = 0
        elif measure_type == Measure.FORWARD:
            B_PX = basis.bond_coeffs(tau=T_fwd-t_)[0]
            CxCxB_P = np.dot(np.dot(C_t, np.transpose(C_t)), B_PX)
            adj_x_drift = -np.einsum('i,j->ji', CxCxB_P, sigma0[:, 0] ** 2)
            betaxCxB_P = np.dot(np.dot(B_PX, C_t), beta_t)
            adj_vol_drift = -sigma0 * betaxCxB_P

            # adj_x_drift = 0
            # adj_vol_drift = 0
        else:
            raise NotImplementedError

        Omega_t = np.tile(Omega_t, (nb_path, 1))
        # multiply by stoch vol driver
        Omega_t = prod_mc(Omega_t, sigma0[:, 0] * sigma0[:, 0])
        # if skew is modeled through DLN skew, not beta, formula for C and Omega are different
        if bxs is not None:
            ys = np.zeros((nb_path, params0.basis.get_nb_factors()))
            tenors = params0.basis.key_terms
            for idx_tenor, tenor in enumerate(tenors):
                ys[:, idx_tenor] = -1.0 / tenor * np.log(
                    params0.basis.bond(t=t_, T=t_ + tenor, x=x0, y=y0, ccy=params0.ccy))
            C_t = params0.calc_factor_vols_dln(yield_vols=A[idx_t], yields=ys, b_dln=bxs, nb_path=nb_path)
            # C_t_2 = params0.calc_factor_vols_dln2(t=t_, yield_vols=A[idx_t], b_dln=bxs)

            Omega_t = np.zeros((nb_path, params0.basis.get_nb_aux_factors()))
            for i, C_t_ith in enumerate(C_t):
                var_t = np.dot(C_t_ith, np.transpose(C_t_ith))
                Omega_t[i, :] = params0.basis.calc_Omega(var_t)

        # make mean 0
        # w0 = w0 - np.average(w0, axis=0)
        # w1 = w1 - np.average(w1, axis=0)

        I0 = I0 + dt * (x0.dot(B0_X) + y0.dot(B0_Y))
        y0 = y0 + dt * (y0.dot(np.transpose(D_Y)) + Omega_t)
        if bxs is not None:
            # if skew is modeled through DLN skew, not beta, formula for C and Omega are different
            # they become stochastic
            for i in range(nb_path):
                x0[i, :] = x0[i, :] + dt * x0[i, :].dot(np.transpose(D_X)) + w0[i, :].dot(np.transpose(C_t[i, :, :])) * \
                           sigma0[i, 0] + adj_x_drift * dt
        else:
            x0 = x0 + dt * x0.dot(np.transpose(D_X)) + prod_mc(w0.dot(np.transpose(C_t)), sigma0[:, 0]) + adj_x_drift * dt
            log_vol = log_vol + ((kappa1 * theta / sigma0) - (kappa1 - kappa2 * theta + 0.5 * vartheta2) - kappa2 * sigma0) * dt + (
                w0.dot(beta_t)).reshape(nb_path, 1) + volvol_t * w1 + adj_vol_drift * dt
        # sigma0 = sigma0 + (kappa1 + kappa2 * sigma0)*(theta-sigma0)*dt + sigma0*(beta_t * w0 + volvol_t * w1) + adj_vol_drift * dt
        sigma0 = np.exp(log_vol)
        if idx + 1 in idx_ttms:
            x0s.append(x0), y0s.append(y0), I0s.append(I0), sigma0s.append(sigma0)

    return x0s, y0s, I0s, sigma0s


def simulate_logsv_futures_MF2(params: MultiFactRateLogSvParams,
                               ttm,
                               t_start,
                               t_end,
                               basis_type,
                               f0: float = None,
                               W: List[np.ndarray] = None,
                               nb_path: int = 100000,
                               seed: int = None):
    """Simulate futures rate F_t under Q^T"""
    # basis_type = "NELSON-SIEGEL"
    sigma0 = params.sigma0
    theta = params.theta
    kappa1, kappa2 = params.kappa1, params.kappa2
    basis = params.basis
    ts = params.beta.ts
    A = params.A
    R = params.R
    C = params.C
    Omega = params.Omega
    betaxs = params.beta.xs
    volvolxs = params.volvol.xs
    ccy = params.ccy

    assert ts.shape[0] > 0 and ts[0] == 0.0
    nb_times = ts.size - 1

    # check inputs
    nb_factors = basis.get_nb_factors()
    assert A.ndim == 2 and A.shape[0] == nb_times and A.shape[1] == nb_factors
    assert betaxs.ndim == 2 and betaxs.shape[0] == nb_times and betaxs.shape[1] == nb_factors
    assert volvolxs.ndim == 1 and volvolxs.shape[0] == nb_times
    assert R.ndim == 2 and R.shape[0] == R.shape[1] == nb_factors
    assert Omega.ndim == 2 and Omega.shape[0] == nb_times and Omega.shape[1] == basis.get_nb_aux_factors()

    sigma0 = sigma0 * np.ones(nb_path)
    # pre-generate random numbers
    if seed is None:
        seed = 16
    np.random.seed(seed)  # fix seed
    nb_steps, dt, grid_t = set_time_grid(ttm=ttm, nb_steps_per_year=720)
    if W is None:
        W0 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path, basis.get_nb_factors()))  # TODO: undo
        W1 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))  # TODO: undo

    log_vol = np.log(sigma0)
    Delta = t_end - t_start
    b1, b2, h1, h2, h0 = futures_conv_adj(t_start, basis_type, params, 0, Delta, ExpansionOrder.ZERO, True, grid_t)

    x0 = np.zeros((1, params.basis.get_nb_factors()))
    y0 = np.zeros((1, params.basis.get_nb_aux_factors()))
    sigma0 = params.sigma0 * np.ones((1, 1))
    expansion_order = ExpansionOrder.FIRST
    if f0 is None:
        f0 = calc_futures_rate(
            ccy=ccy, basis_type=basis_type, params=params,
            x0=x0, y0=y0, sigma0=sigma0,
            t0=0.0, ttm=ttm, t_start=t_start, t_end=t_end,
            Delta=Delta, expansion_order=expansion_order)[0][0]


    zeta0 = np.log(f0 + 1.0 / Delta)

    a, eta, _, _, _, beta, volvol = params.transform_QT_params(expiry=ttm, t_start=t_start, t_end=t_end, t_grid=grid_t)

    for idx, (t_, w0, w1) in enumerate(zip(grid_t, W0, W1)):
        # interpolation of the volatility matrix is tedious
        idx_t = bracket(ts[1:], t_, throw_if_not_found=True)
        beta_t = betaxs[idx_t]
        volvol_t = volvolxs[idx_t]
        # C_t = C[idx_t]
        # Omega_t = Omega[idx_t]
        vartheta2 = np.dot(beta_t, beta_t) + volvol_t * volvol_t

        # Omega_t = np.tile(Omega_t, (nb_path, 1))
        # multiply by stoch vol driver
        # Omega_t = prod_mc(Omega_t, sigma0[:, 0] * sigma0[:, 0])

        tau_start = t_start - t_
        tau_end = t_end - t_
        tau_exp = ttm - t_
        B_P_start = basis.bond_coeffs(tau_start)[0]
        B_P_end = basis.bond_coeffs(tau_end)[0]
        B_P_exp = basis.bond_coeffs(tau_exp)[0]
        h1_t = h1[idx]

        a_t = a[idx]
        eta_t = eta[idx]
        # kappa0_t = kappa0[idx]
        # kappa1_t = kappa1[idx]
        # kappa2_t = kappa2[idx]

        a0_t = a_t + beta_t*h1_t
        a1_t = volvol_t * h1_t
        # eta_t = np.dot(np.transpose(C_t), B_P_exp)
        # delta_t = np.dot(a0_t, eta_t)
        adj_vol_drift = np.dot(beta_t, eta_t)  # we put plus so that adjustment is additive to kappa2
        zeta0 = zeta0 + (-np.dot(a0_t, eta_t) - 0.5 * np.dot(a0_t, a0_t) - 0.5 * a1_t * a1_t) * sigma0 * sigma0 * dt + sigma0 * w0.dot(
            a0_t) + sigma0 * w1 * a1_t
        log_vol = log_vol + ((kappa1*theta/sigma0) - (kappa1 - kappa2*theta + 0.5 * vartheta2) - (kappa2 + adj_vol_drift) * sigma0) * dt + w0.dot(beta_t) + volvol_t * w1
        sigma0 = np.exp(log_vol)
        f0 = np.exp(zeta0) - 1.0 / Delta

    return f0


def simulate_logsv_futures_MF(ttm: float,
                              sigma0: np.ndarray,
                              theta: float,
                              kappa1: float,
                              kappa2: float,
                              ts: np.ndarray,
                              A: List[np.ndarray],  # for multi-factor model
                              R: List[np.ndarray],  # for multi-factor model
                              B: List[np.ndarray],  # for multi-factor model
                              C: List[np.ndarray],  # for multi-factor model
                              Omega: List[np.ndarray],  # for multi-factor model
                              h1: float,  # coefficient from futures conv adj
                              betaxs: np.ndarray,
                              volvolxs: np.ndarray,
                              basis: NelsonSiegel,
                              t_start: float,
                              t_end: float,
                              ccy: str,
                              f0: float = None,
                              nb_path: int = 100000,
                              seed: int = None,
                              W: List[np.ndarray] = None
                              ) -> np.ndarray:
    assert ts.shape[0] > 0 and ts[0] == 0.0
    nb_times = ts.size - 1

    # check inputs
    nb_factors = basis.get_nb_factors()
    assert A.ndim == 3 and A.shape[0] == nb_times and A.shape[1] == A.shape[2] == nb_factors
    assert betaxs.ndim == 2 and betaxs.shape[0] == nb_times and betaxs.shape[1] == nb_factors
    assert volvolxs.ndim == 1 and volvolxs.shape[0] == nb_times
    assert R.ndim == 2 and R.shape[0] == R.shape[1] == nb_factors and np.all(R.shape == B.shape)
    assert Omega.ndim == 2 and Omega.shape[0] == nb_times and Omega.shape[1] == basis.get_nb_aux_factors()

    if sigma0.shape[0] == 1:
        sigma0 = sigma0 * np.ones(nb_path)
    else:
        assert sigma0.shape[0] == nb_path

    # pre-generate random numbers
    if seed is None:
        seed = 16
    np.random.seed(seed)  # fix seed
    nb_steps, dt, grid_t = set_time_grid(ttm=ttm)
    if W is None:
        W0 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path, basis.get_nb_factors()))  # TODO: undo
        W1 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))  # TODO: undo
    else:
        W0 = W[0] * np.sqrt(dt)
        W1 = W[1] * np.sqrt(dt)

    log_vol = np.log(sigma0)
    x0 = np.zeros((nb_path, basis.get_nb_factors(),))
    y0 = np.zeros((nb_path, basis.get_nb_aux_factors(),))
    # In case if we do not provide futures rate as an input, replace it by forward rate
    if f0 is None:
        f0 = basis.libor_rate(t=0, t_start=t_start, t_end=t_end, x=x0, y=y0, ccy=ccy)
    delta_frac = t_end - t_start
    zeta0 = np.log(f0 + 1.0/delta_frac)

    for idx, (t_, w0, w1) in enumerate(zip(grid_t, W0, W1)):
        # w1 = np.reshape(w1, (nb_path, 1))
        # interpolation of the volatility matrix is tedious
        idx_t = bracket(ts[1:], t_, throw_if_not_found=True)
        beta_t = betaxs[idx_t]
        volvol_t = volvolxs[idx_t]
        C_t = C[idx_t]
        vartheta2 = np.dot(beta_t, beta_t) + volvol_t * volvol_t
        tau_start = t_start - t_
        tau_end = t_end - t_
        tau_exp = ttm - t_
        B_P_start = basis.bond_coeffs(tau_start)[0]
        B_P_end = basis.bond_coeffs(tau_end)[0]
        B_P_exp = basis.bond_coeffs(tau_exp)[0]

        a_t = np.dot(np.transpose(C_t), B_P_end - B_P_start)
        eta_t = np.dot(np.transpose(C_t), B_P_exp)
        delta_t = np.dot(a_t, eta_t)
        adj_vol_drift = -np.dot(beta_t, eta_t) * sigma0
        zeta0 = zeta0 + (-delta_t - 0.5 * np.dot(a_t, a_t)) * sigma0 * sigma0 * dt + sigma0 * w0.dot(a_t)

        log_vol = log_vol + ((kappa1 * theta / sigma0) - (
                kappa1 - kappa2 * theta + 0.5 * vartheta2) - kappa2 * sigma0) * dt + w0.dot(
            beta_t) + volvol_t * w1 + adj_vol_drift * dt
        sigma0 = np.exp(log_vol)
        f0 = np.exp(zeta0) - 1.0 / delta_frac

    return f0


# @njit(cache=False, fastmath=True)  # TODO
def simulate_logsv_swap_approx_terminal(ttm: float,
                                        sigma0: np.ndarray,
                                        theta: float,
                                        kappa1: float,
                                        kappa2: float,
                                        ts: np.ndarray,
                                        axs: np.ndarray,
                                        betaxs: np.ndarray,
                                        volvolxs: np.ndarray,
                                        t0: float = 0.0,
                                        s0: np.ndarray = None,
                                        seed: float = None,
                                        nb_path: int = 100000
                                        ) -> np.ndarray:
    if sigma0.shape[0] == 1:
        sigma0 = sigma0 * np.ones(nb_path)
    else:
        assert sigma0.shape[0] == nb_path

    # pre-generate random numbers
    if seed is None:
        seed = 16
    # pre-generate random numbers
    np.random.seed(seed)  # fix seed
    nb_steps, dt, grid_t = set_time_grid(ttm=ttm)
    grid_t = t0 + grid_t
    W0 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))
    W1 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))

    log_vol = np.log(sigma0)
    ts_sw = get_default_swap_term_structure(t0+ttm)
    if s0 is None:
        s0 = np.array([swap_rate(0, ts_sw=ts_sw, x=0, y=0)[0]])
    if s0.shape[0] == 1:
        s0 = s0 * np.ones(nb_path)
    swap_der = swap_rate(grid_t, ts_sw=ts_sw, x=0, y=0)[1]

    for t_, w0, w1 in zip(grid_t, W0, W1):
        a_t = pw_const(ts, axs, t_, False)
        beta_t = pw_const(ts, betaxs, t_, False)
        volvol_t = pw_const(ts, volvolxs, t_, False)
        vartheta2 = beta_t * beta_t + volvol_t * volvol_t

        ds_dx = swap_rate(t=t_, ts_sw=ts_sw, x=0, y=0)[1]
        ann_der0_mc = annuity(t=t_, ts_sw=ts_sw, x=0, y=0, m=0)
        ann_der1_mc = annuity(t=t_, ts_sw=ts_sw, x=0, y=0, m=1)
        d_loga_dx = ann_der1_mc / ann_der0_mc
        adj_vol_drift = a_t * beta_t * sigma0 * d_loga_dx

        s0 = s0 + ds_dx * a_t * sigma0 * w0
        log_vol = log_vol + ((kappa1 * theta / sigma0) - (
                kappa1 - kappa2 * theta + 0.5 * vartheta2) - kappa2 * sigma0) * dt + beta_t * w0 + volvol_t * w1 + adj_vol_drift * dt
        sigma0 = np.exp(log_vol)

    return s0


# @njit(cache=False, fastmath=True)  # TODO
def compute_mcapprox_payoff(ttm: float,
                            s_mc: np.ndarray,
                            strikes_ttm: np.ndarray,
                            optiontypes_ttm: np.ndarray):
    payoffsign = np.where(optiontypes_ttm == 'P', -1, 1).astype(float)
    option_prices = np.zeros_like(strikes_ttm)
    option_std = np.zeros_like(strikes_ttm)

    for idx, (strike, sign) in enumerate(zip(strikes_ttm, payoffsign)):
        option_prices[idx] = np.nanmean(np.maximum(sign * (s_mc - strike), 0))
        option_std[idx] = np.nanstd(np.maximum(sign * (s_mc - strike), 0))

    return option_prices, option_std / np.sqrt(s_mc.shape[0])


# @njit(cache=False, fastmath=True)  # TODO
def calculate_swap_rate_terminal(ttm: float,
                                 x0: np.ndarray,
                                 y0: np.ndarray,
                                 I0: np.ndarray,
                                 ts_sw: np.ndarray):
    # calculate swap rates using simulated states
    # ts_sw = get_default_swap_term_structure(ttm)
    s_mc = swap_rate(t=ttm, ts_sw=ts_sw, x=x0, y=y0, is_mc_mode=True)[0]
    ann_mc = annuity(t=ttm, ts_sw=ts_sw, x=x0, y=y0, m=0, is_mc_mode=True)
    numer = 1.0 / bond(t=0, T=ttm, x=0, y=0, m=0, is_mc_mode=True) * np.exp(I0)
    return ts_sw, s_mc, ann_mc, numer


def calculate_swap_rate(t: float,
                        ttm: float,
                        x0: np.ndarray,
                        y0: np.ndarray,
                        I0: np.ndarray,
                        ts_sw: np.ndarray = None):
    if ts_sw is None:
        ts_sw = get_default_swap_term_structure(ttm)
    # calculate swap rates using simulated states
    s_mc = swap_rate(t=t, ts_sw=ts_sw, x=x0, y=y0, is_mc_mode=True)[0]
    ann_mc = annuity(t=t, ts_sw=ts_sw, x=x0, y=y0, m=0, is_mc_mode=True)
    numer = 1.0 / bond(t=0, T=t, x=0, y=0, m=0, is_mc_mode=True) * np.exp(I0)
    return ts_sw, s_mc, ann_mc, numer


# @njit(cache=False, fastmath=True)  # TODO
def compute_mc_vars_payoff(ttm: float,
                           x0: np.ndarray,
                           y0: np.ndarray,
                           I0: np.ndarray,
                           strikes_ttm: np.ndarray,
                           optiontypes_ttm: np.ndarray,
                           is_annuity_measure: bool = False):
    ts_sw, s_mc, ann_mc, numer = calculate_swap_rate_terminal(ttm, x0, y0, I0)
    payoffsign = np.where(optiontypes_ttm == 'P', -1, 1).astype(float)
    option_prices = np.zeros_like(strikes_ttm)
    option_std = np.zeros_like(strikes_ttm)

    df = bond(0, ttm, 0, 0, 0, False)
    ann_crv = annuity(ttm, ts_sw, 0, 0, 0)
    s0_crv = swap_rate(ttm, ts_sw, 0, 0)[0]
    # numer = numer * df

    for idx, (strike, sign) in enumerate(zip(strikes_ttm, payoffsign)):
        if is_annuity_measure:
            option_prices[idx] = np.nanmean(np.maximum(sign * (s_mc - strike), 0))
            option_std[idx] = np.nanstd(np.maximum(sign * (s_mc - strike), 0))
        else:
            option_prices[idx] = np.nanmean(1. / numer * ann_mc * np.maximum(sign * (s_mc - strike), 0)) / ann_crv / df
            option_std[idx] = np.nanstd(1. / numer * ann_mc * np.maximum(sign * (s_mc - strike), 0)) / ann_crv / df

    return option_prices, option_std / np.sqrt(x0.shape[0])
