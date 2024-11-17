import copy
import numpy as np
import pandas as pd
import seaborn as sns
from dataclasses import dataclass
from enum import Enum
from typing import Union, Dict, Tuple, Optional
from numba.typed import List
import matplotlib.pyplot as plt

import stochvolmodels.pricers.analytic.bachelier as bachel
from stochvolmodels import LogSvParams
from stochvolmodels.pricers.factor_hjm.rate_evaluate import libor_rate
from stochvolmodels.data.option_chain import FutOptionChain, SwOptionChain
from stochvolmodels.pricers.factor_hjm.rate_logsv_params import MultiFactRateLogSvParams, TermStructure
from stochvolmodels.pricers.factor_hjm.rate_factor_basis import NelsonSiegel
from stochvolmodels.pricers.factor_hjm.rate_logsv_pricer import RateFutLogSVPricer
from stochvolmodels.pricers.factor_hjm.rate_affine_expansion import UnderlyingType
from stochvolmodels.pricers.factor_hjm.rate_core import generate_ttms_grid
from stochvolmodels.pricers.logsv.affine_expansion import ExpansionOrder
from stochvolmodels.pricers.factor_hjm.rate_core import get_futures_start_and_pmt
from stochvolmodels.pricers.factor_hjm.rate_logsv_pricer import  logsv_chain_de_pricer, Measure, calc_futures_rate, FutSettleType, simulate_logsv_futures_MF2
from stochvolmodels.pricers.factor_hjm.rate_logsv_ivols import get_delta_at_strikes, infer_strikes_from_deltas, calc_logsv_ivols, fit_logsv_ivols
from stochvolmodels.pricers.factor_hjm.factor_hjm_pricer import do_mc_simulation

def getFutCalibRateLogSVParams(type_str: str = "NELSON-SIEGEL") -> Dict[str, MultiFactRateLogSvParams]:
    """return dictionary of parameters for rate future options, currency is USD"""
    dict = {}

    # ttms = np.array([28.0, 56.0, 119.0]) / 365.0
    ttms = np.array([
        # 31.0,
        75.0, 103.0]) / 365.0
    # ttms = np.array([40.0, 103.0, 194.0]) / 365.0
    R_corr = np.array([[1.0, 0.99, 0.97], [0.99, 1.0, 0.98], [0.97, 0.98, 1.0]])

    if type_str == "NELSON-SIEGEL":
        nelson_siegel = NelsonSiegel(meanrev=0.55, key_terms=np.array([2.0, 5.0, 10.0]))
        nb_factors = NelsonSiegel.get_nb_factors()
        times = np.concatenate((0, ttms), axis=None)

        params0 = MultiFactRateLogSvParams(
            sigma0=1.0, theta=1.0, kappa1=1e-12, kappa2=1e-12,
            beta=TermStructure.create_multi_fact_from_vec(times, RateFutLogSVPricer.populate_betas(1e-12, basis=nelson_siegel)),
            volvol=TermStructure.create_from_scalar(times, 1e-12),
            A=np.array([0.01, 0.01, 0.01]),
            R=R_corr,
            basis=nelson_siegel,
            ccy="USD_NS", vol_interpolation="BY_YIELD")


        # test gaissian case
        params0.update_params(idx=0, kappa1=0.5, kappa2=1.0)
        params0.update_params(idx=0, A_idx=np.array([0.999, 0.626, 0.009])*0.01,
                              beta_idx=RateFutLogSVPricer.populate_betas(-0.567, basis=nelson_siegel), volvol_idx=1.398)
        params0.update_params(idx=1, A_idx=np.array([1.316, 1.342, 0.795])*0.01,
                              beta_idx=RateFutLogSVPricer.populate_betas(-0.928, basis=nelson_siegel), volvol_idx=0.564)
        #

        dict["USD_NS"] = params0
        dict["USD"] = params0

    else:
        raise NotImplementedError
    return dict

def get_futures_data(atm_ranges: np.ndarray = None) -> FutOptionChain:
    strks = [[94.625, 94.6875, 94.75, 94.8125, 94.875, 94.9375, 95, 95.0625, 95.125, 95.1875, 95.25, 95.3125, 95.375, 95.4375, 95.5, 95.5625, 95.625],
              [94.625, 94.6875, 94.75, 94.8125, 94.875, 94.9375, 95, 95.0625, 95.125, 95.1875, 95.25, 95.3125, 95.375, 95.4375, 95.5]]
    vols = [[85.18, 83.53, 80.65, 80.02, 80.25, 78.44, 80.68, 81.98, 85.46, 86.45, 88.14, 90.08, 89.86, 91.92, 93.65, 95.21, 100.49],
             [81.31, 78.6, 77.92, 78.22, 78.83, 79.42, 81.29, 82.63, 84.77, 86.39, 87.66, 89.3, 90.52, 91.5, 93.63]]
    fut_rates = np.array([95.25, 95.25])
    ttms = np.array([75.0, 103.0]) / 365
    fwds = np.array([libor_rate(0, ttm, ttm+0.25, 0, 0) for ttm in ttms])
    # assert np.all([strks_ttm[2] == fut_rate for strks_ttm, fut_rate in zip(strks, fut_rates)])
    strks = [np.array(strks_ttm) - fut_rate + (100 - 100*fwd) for strks_ttm, fut_rate, fwd in zip(strks, fut_rates, fwds)]

    chain = FutOptionChain(ccy="USD_NS",
                           ttms=ttms,
                           forwards=fwds,
                           strikes_ttms=[(100 - strks_ttm) * 0.01 for strks_ttm in strks],
                           ivs_call_ttms=[np.array(vols_ttm) * 0.0001 for vols_ttm in vols],
                           ivs_put_ttms=[np.array(vols_ttm) * 0.0001 for vols_ttm in vols],
                           ttms_ids=np.array(['75d', '103d']),
                           call_oi=None,
                           put_oi=None,
                           ticker='DUMMY')
    return chain

def refit_to_sabr(futoption_chain: FutOptionChain) -> (FutOptionChain,
                                                       Dict[str, np.ndarray]):
    calib_params = {'alpha': np.zeros_like(futoption_chain.ttms),
                    'beta': np.zeros_like(futoption_chain.ttms),
                    'total_vol': np.zeros_like(futoption_chain.ttms),
                    'rho': np.zeros_like(futoption_chain.ttms)}

    ivols_opt_ttms = []
    strikes_opt_ttms = []

    for idx_ttm, ttm in enumerate(futoption_chain.ttms):
        beta = 0.0  # normal SABR
        shift = 0.0  # non-shifted SABR
        calib_param = fit_logsv_ivols(strikes=futoption_chain.strikes_ttms[idx_ttm],
                                      mid_vols=futoption_chain.ivs_call_ttms[idx_ttm],
                                      f0=futoption_chain.forwards[idx_ttm],
                                      beta=0.0,
                                      shift=0.0,
                                      ttm=ttm)
        calib_params['alpha'][idx_ttm] = calib_param['alpha']
        calib_params['beta'][idx_ttm] = calib_param['beta']
        calib_params['total_vol'][idx_ttm] = calib_param['total_vol']
        calib_params['rho'][idx_ttm] = calib_param['rho']

        # re-calculate strikes
        delta_grid = np.array([-0.25, -0.375, -0.5, 0.375, 0.25])
        f0 = futoption_chain.forwards[idx_ttm]
        alpha = calib_param['alpha']

        # check deltas
        deltas = get_delta_at_strikes(strikes=futoption_chain.strikes_ttms[idx_ttm], f0=f0, ttm=ttm, sigma0=alpha,
                                      rho=calib_param['rho'],
                                      total_vol=calib_param['total_vol'], beta=beta, shift=shift)
        df = pd.DataFrame.from_dict(dict(zip(["STRIKE", "IV", "DELTA"], [futoption_chain.strikes_ttms[idx_ttm], futoption_chain.ivs_call_ttms[idx_ttm], deltas])))
        # print(df)
        strikes_grid = infer_strikes_from_deltas(deltas=delta_grid, f0=f0, ttm=ttm, sigma0=alpha,
                                                 rho=calib_param['rho'],
                                                 total_vol=calib_param['total_vol'], beta=beta, shift=shift).values

        ivols_opt = calc_logsv_ivols(strikes=strikes_grid,
                                     f0=futoption_chain.forwards[idx_ttm],
                                     ttm=ttm,
                                     alpha=calib_param['alpha'],
                                     rho=calib_param['rho'],
                                     total_vol=calib_param['total_vol'],
                                     beta=beta,
                                     shift=shift)
        ivols_opt_ttms.append(ivols_opt)
        strikes_opt_ttms.append(strikes_grid)

    # print(calib_params)

    futoption_chain_sabr = FutOptionChain(ccy=futoption_chain.ccy,
                                          ttms=futoption_chain.ttms,
                                          forwards=futoption_chain.forwards,
                                          strikes_ttms=np.array(strikes_opt_ttms),
                                          ivs_call_ttms=np.array(ivols_opt_ttms),
                                          ivs_put_ttms=np.array(ivols_opt_ttms),
                                          ttms_ids=futoption_chain.ttms_ids,
                                          call_oi=None,
                                          put_oi=None,
                                          ticker=futoption_chain.ticker)

    return futoption_chain_sabr, calib_params


def plot_mkt_model_joint_fut_smile_MF(params0: MultiFactRateLogSvParams,
                                      ttms: np.ndarray,
                                      ttms_ids: List[str],
                                      forward_ttms: np.ndarray,
                                      normal_vols_ttms: np.ndarray,
                                      strikes_ttms: np.ndarray,
                                      optiontypes: np.ndarray,
                                      is_stiff_solver: bool,
                                      expansion_order: ExpansionOrder,
                                      slice_ids: List[str] = None,
                                      plot_market: bool = True,
                                      add_up_down: bool = False) -> plt.Figure:
    # strikes and vols must have same dimensions
    assert normal_vols_ttms.shape == strikes_ttms.shape
    # expiries and their ids must be consistent. check forwards
    assert len(ttms) == len(ttms_ids) == forward_ttms.size and forward_ttms.ndim == 1
    # number of strikes must be consistent with expiries
    assert strikes_ttms.ndim == 2 and strikes_ttms.shape[0] == len(ttms_ids)
    # types and strikes must have same size
    assert optiontypes.ndim == 1 and optiontypes.size == strikes_ttms.shape[1]

    ccy = "USD"
    nb_cols = len(ttms_ids)


    t_grid = generate_ttms_grid(ttms)
    # palettes = ['blue', 'green', 'magenta', 'cyan', 'orange']
    ticksizes = np.asarray([0.25, 0.25, 0.5]) * 0.0001   # first two expiries: quarter of bp, last is half bp

    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(1, nb_cols, figsize=(18, 5), tight_layout=True)

    for idx_ttm, ttm in enumerate(ttms):
        forward0 = forward_ttms[idx_ttm]
        strikes_ttm = strikes_ttms[idx_ttm]
        normal_vols_ttm = normal_vols_ttms[idx_ttm]
        ttm_ = np.array([ttm])

        model_prices_ttms, model_ivs_ttms = logsv_chain_de_pricer(params=params0,
                                                                  t_grid=t_grid,
                                                                  ttms=ttm_,
                                                                  forwards=[[forward0]],
                                                                  strikes_ttms=[[strikes_ttm]],
                                                                  optiontypes_ttms=[optiontypes],
                                                                  do_control_variate=False,
                                                                  is_stiff_solver=is_stiff_solver,
                                                                  expansion_order=expansion_order,
                                                                  underlying_type=UnderlyingType.FUTURES,
                                                                  lag=0,  # TODO: review it for mid-curve options
                                                                  x0=None,
                                                                  y0=None)

        if add_up_down:
            ticksize = ticksizes[idx_ttm]
            option_up = np.ones_like(normal_vols_ttm)
            option_down = np.ones_like(normal_vols_ttm)
            for idx, (strike0, vol0, type0) in enumerate(zip(strikes_ttm, normal_vols_ttm, optiontypes)):
                pv0 = bachel.compute_normal_price(forward=forward0, strike=strike0, ttm=ttm, vol=vol0, optiontype=type0)
                option_up[idx] = pv0 + ticksize
                option_down[idx] = np.maximum(pv0 - ticksize, 0.0)

            mkt_ivols_up = bachel.infer_normal_ivols_from_slice_prices(ttm=ttm, forward=forward0,
                                                                       strikes=strikes_ttm,
                                                                       model_prices=option_up,
                                                                       optiontypes=optiontypes,
                                                                       discfactor=1.0)
            mkt_ivols_down = bachel.infer_normal_ivols_from_slice_prices(ttm=ttm, forward=forward0,
                                                                         strikes=strikes_ttm,
                                                                         model_prices=option_down,
                                                                         optiontypes=optiontypes,
                                                                         discfactor=1.0)

        headers = ('(A)', '(B)', '(C)', '(D)', '(E)', '(F)')
        if ttms.size > 6:
            raise NotImplementedError(f"Extend header tags")
        else:
            headers = headers[:ttms.size]
        ax = axs[idx_ttm] if len(ttms) > 1 else axs
        x_grid = bachel.strikes_to_delta(strikes=strikes_ttm,
                                  ivols=normal_vols_ttm,
                                  f0=forward0,
                                  ttm=ttm)
        mkt_ivols = pd.Series(normal_vols_ttm, index=x_grid, name=f"market").sort_index()
        mkt_ivols = SwOptionChain.remap_to_inc_delta(mkt_ivols)
        model_ivols = pd.Series(model_ivs_ttms[0][0], index=x_grid, name=f"model").sort_index()
        model_ivols = SwOptionChain.remap_to_inc_delta(model_ivols)
        # although strikes to delta mapping must be recalculated, we ignore it in our plot
        mkt_ivols_up = pd.Series(mkt_ivols_up, index=x_grid, name=f"up").sort_index()
        mkt_ivols_up = SwOptionChain.remap_to_inc_delta(mkt_ivols_up)
        mkt_ivols_down = pd.Series(mkt_ivols_down, index=x_grid, name=f"down").sort_index()
        mkt_ivols_down = SwOptionChain.remap_to_inc_delta(mkt_ivols_down)

        xvar_format = '{:.2%}'
        sns.lineplot(data=pd.concat([model_ivols], axis=1), ax=ax, palette=['blue'], markers=False)
        if plot_market:
            sns.scatterplot(data=pd.concat([mkt_ivols], axis=1), ax=ax, palette=['orange'])
            if add_up_down:
                sns.scatterplot(data=pd.concat([mkt_ivols_up], axis=1), ax=ax, palette=['green'], markers=[7])
                sns.scatterplot(data=pd.concat([mkt_ivols_down], axis=1), ax=ax, palette=['red'], markers=[6])
        # report_ax_bps(ax=ax, xticks=swaption_chain.strikes_ttms[idx], y_axis_in_bps=True,
        #               y_label="Implied normal vols (bp)")
        # print(model_ivs_ttms)

        ax.set_xticks([-0.8, -0.65, -0.5, -0.35, -0.2])
        ax.set_xticklabels(['{:.2f}'.format(x, 2) for x in SwOptionChain.remap_to_pc_delta(ax.get_xticks())])
        ax.set_yticklabels(['{:.0f}'.format(x * 10000, 2) for x in ax.get_yticks()])
        title = f"{ccy}: {ttms_ids[idx_ttm]} market data"
        ax.set_title(title, color='darkblue')
        h, l = ax.get_legend_handles_labels()
        ax.legend(loc="upper right")

    return fig


def calc_mc_vols(basis_type: str,
                 params: MultiFactRateLogSvParams,
                 ttm: float,
                 lag: float,
                 forward: float,
                 strikes: np.ndarray,
                 optiontypes: np.ndarray,
                 measure_type: Measure,
                 nb_path: int,
                 T_fwd: float = None,
                 x0: np.ndarray = None,
                 y0: np.ndarray = None,
                 sigma0: np.ndarray = None,
                 I0: np.ndarray = None,
                 seed: int = None) -> (List[np.ndarray], List[np.ndarray]):

    t_start, t_end = get_futures_start_and_pmt(t0=ttm, lag=lag)
    Delta = t_end - t_start
    ttms = np.array([ttm])
    forwards = np.array([forward])
    # we simulate under risk-neutral measure only
    if x0 is None:
        x0 = np.zeros((nb_path, params.basis.get_nb_factors()))
    else:
        assert x0.shape == (nb_path, params.basis.get_nb_factors(),)
    if y0 is None:
        y0 = np.zeros((nb_path, params.basis.get_nb_aux_factors()))
    else:
        assert y0.shape == (nb_path, params.basis.get_nb_aux_factors(),)
    if sigma0 is None:
        sigma0 = np.ones((nb_path, 1))
    else:
        assert sigma0.shape == (nb_path, 1)
    if I0 is None:
        I0 = np.zeros((nb_path,))
    else:
        assert I0.shape == (nb_path,)

    sim_factor = True
    if sim_factor:
        # do simulation under risk-neutral measure
        x0s, y0s, I0s, _ = do_mc_simulation(basis_type=basis_type,
                                            ccy=params.ccy,
                                            ttms=ttms,
                                            x0=x0,
                                            y0=y0,
                                            I0=I0,
                                            sigma0=sigma0,
                                            params=params,
                                            nb_path=nb_path,
                                            seed=seed,
                                            ts_sw=None,
                                            measure_type=measure_type,
                                            T_fwd=T_fwd)
        x0 = x0s[-1]
        y0 = y0s[-1]
        I0 = I0s[-1]

        # we calculate futures at settlement
        assert np.isclose(t_start, ttm)
        P_Ts_Te = params.basis.bond(t=t_start, T=t_end, x=x0, y=y0, ccy=params.ccy, m=0)
        libor_Ts_Te = 1.0 / Delta * (1.0 / P_Ts_Te - 1.0)
        f_mc = libor_Ts_Te
        zeta_T = np.log(libor_Ts_Te + 1.0 / Delta)
#         zeta_0 = np.log(f0 + 1.0 / Delta)


    else:
        f_mc = simulate_logsv_futures_MF2(params, ttm, t_start, t_end, basis_type=basis_type,
                                          f0=forward, nb_path=nb_path, seed=seed)

    # df = params.basis.bond(t=0, T=ttm, x=x0[0,:], y=y0[0,:], m=0, ccy=params.ccy)[0]
    # print(f"Mean f_mc: {np.mean(f_mc):.4f}, f0: {forward:.4f}")
    df = 1.0
    mc_vols = List()
    mc_prices = List()
    mc_vols_ups = List()
    mc_vols_downs = List()
    std_factor = 1.96

    # calculate option payoffs
    payoffsign = np.where(optiontypes == 'P', -1, 1).astype(float)
    option_mean = np.zeros_like(strikes)
    option_std = np.zeros_like(strikes)

    for idx, (strike, sign) in enumerate(zip(strikes, payoffsign)):
        option_mean[idx] = np.nanmean( df * np.maximum(sign * (f_mc - strike), 0)) # /  bond0
        option_std[idx] = np.nanstd( df * np.maximum(sign * (f_mc - strike), 0)) # / bond0
        option_std[idx] = option_std[idx] / np.sqrt(nb_path)

    option_up = option_mean + std_factor * option_std
    option_down = np.maximum(option_mean - std_factor * option_std, 0.0)

    mc_ivols_mid = bachel.infer_normal_ivols_from_chain_prices(ttms=ttms,
                                                                     forwards=forwards,
                                                                     discfactors=np.ones_like(ttms)*df,
                                                                     strikes_ttms=[strikes],
                                                                     optiontypes_ttms=[optiontypes],
                                                                     model_prices_ttms=[option_mean])
    mc_ivols_up = bachel.infer_normal_ivols_from_chain_prices(ttms=ttms,
                                                                    forwards=forwards,
                                                                    discfactors=np.ones_like(ttms)*df,
                                                                    strikes_ttms=[strikes],
                                                                    optiontypes_ttms=[optiontypes],
                                                                    model_prices_ttms=[option_up])
    mc_ivols_down = bachel.infer_normal_ivols_from_chain_prices(ttms=ttms,
                                                                      forwards=forwards,
                                                                      discfactors=np.ones_like(ttms)*df,
                                                                      strikes_ttms=[strikes],
                                                                      optiontypes_ttms=[optiontypes],
                                                                      model_prices_ttms=[option_down])

    mc_vols.append(mc_ivols_mid[0])
    mc_vols_ups.append(mc_ivols_up[0])
    mc_vols_downs.append(mc_ivols_down[0])


    return mc_prices, mc_vols, mc_vols_ups, mc_vols_downs


class UnitTests(Enum):
    CALIBRATE_LOGSV_FUT = 6
    BENCHMARK_ANALYTIC_VS_MC_FUT = 12


def run_unit_test(unit_test: UnitTests):
    if unit_test == UnitTests.CALIBRATE_LOGSV_FUT:
        futoption_chain_fit = get_futures_data()
        futoption_chain_fit, _ = refit_to_sabr(futoption_chain=futoption_chain_fit)

        params0 = getFutCalibRateLogSVParams(type_str="NELSON-SIEGEL")["USD"]
        params0.q = params0.theta * 1.0

        ttms = futoption_chain_fit.ttms
        assert np.all(ttms == params0.ts[1:])

        is_stiff_solver = True
        expansion_order = ExpansionOrder.FIRST
        # if we calibrate the model parameters to option market data
        logsv_pricer = RateFutLogSVPricer()


        # alternatively we can choose parameters manually and plot
        opt_params = copy.deepcopy(params0)
        # opt_params = opt_params.reduce(['84d', '175d'])
        opt_params = opt_params.reduce(['75d', '103d'])

        # print(opt_params)

        for ttm in opt_params.ts[1:]:
            assert opt_params.check_QT_kappa2(t_start=ttm)


        fig = plot_mkt_model_joint_fut_smile_MF(params0=opt_params,
                                                ttms=ttms,
                                                ttms_ids=futoption_chain_fit.ttms_ids.tolist(),
                                                forward_ttms=futoption_chain_fit.forwards,
                                                normal_vols_ttms=futoption_chain_fit.ivs_call_ttms,
                                                strikes_ttms=futoption_chain_fit.strikes_ttms,
                                                optiontypes=futoption_chain_fit.optiontypes_ttms[0],
                                                slice_ids=None,
                                                plot_market=True,
                                                add_up_down=True,
                                                expansion_order=expansion_order,
                                                is_stiff_solver=is_stiff_solver)

        # fig.savefig(f"..//draft//figures//fhjm//calibration_FHJM_futures.pdf")

    elif unit_test == UnitTests.BENCHMARK_ANALYTIC_VS_MC_FUT:
        mult = 1.0
        curr = "USD"
        for seed in [20]:

            futoption_chain_fit = get_futures_data()  # FUTURES OPTION DATA
            futoption_chain_fit, _ = refit_to_sabr(futoption_chain=futoption_chain_fit)
            basis_type = "NELSON-SIEGEL"
            params0 = getFutCalibRateLogSVParams(type_str=basis_type)[curr]
            # params0 = params0.reduce(['84d'])
            params0.q = params0.theta * mult
            # check condition on eigenvalues for model parameters
            for n in range(len(params0.ts[1:])):
                logsv_params = LogSvParams(sigma0=params0.sigma0, theta=params0.theta, kappa1=params0.kappa1, kappa2=params0.kappa2,
                                           beta=0.0, volvol=np.sqrt(np.dot(params0.beta.xs[n], params0.beta.xs[n]) + params0.volvol.xs[n] ** 2))
                logsv_params.assert_vol_moments_stability(n_terms=4)


            # assert ttm_id in futoption_chain_fit.ttms_ids
            futoption_chain_fit2 = futoption_chain_fit.reduce_ttms(['75d'])
            ttms = futoption_chain_fit2.ttms
            # assert ttms.size == 1

            # re-center strikes
            tenor = 0.25
            t_start, t_end = get_futures_start_and_pmt(t0=ttms[-1], lag=0.0, libor_tenor=tenor)
            f0 = calc_futures_rate(
                ccy=params0.ccy, basis_type=basis_type, params=params0,
                x0=np.zeros((params0.basis.get_nb_factors())),
                y0=np.zeros((params0.basis.get_nb_aux_factors())),
                sigma0=np.ones((1, 1)),
                t0=0.0, t_start=t_start, t_end=t_end, Delta=0.25,
                expansion_order=ExpansionOrder.ZERO, settlement_type=FutSettleType.SOFR)[0][0]
            for idx_ttm, ttm in enumerate(futoption_chain_fit2.ttms):
                futoption_chain_fit2.strikes_ttms[idx_ttm] = futoption_chain_fit2.strikes_ttms[idx_ttm] - futoption_chain_fit2.forwards[idx_ttm] + f0
                futoption_chain_fit2.forwards[idx_ttm] = f0

            futoption_chain_fit2.optiontypes_ttms = np.array(['P', 'P', 'P', 'C', 'C'])

            nb_path = 2**17

            t_grid = generate_ttms_grid(ttms)

            model_ivs_ttms = []
            mc_ivols_ttms = []
            mc_ivols_ups_ttms = []
            mc_ivols_downs_ttms = []

            strikes_ttms_mc = [np.linspace(strikes_ttm[0], strikes_ttm[-1], 21) for strikes_ttm in futoption_chain_fit2.strikes_ttms]
            # strikes_ttms_mc = futoption_chain_fit2.strikes_ttms.copy()
            # assert ttms.size == 1
            assert np.all((futoption_chain_fit2.forwards - f0) <= 1e-6)

            is_stiff_solvers = [True, True]

            params_mc = copy.deepcopy(params0)

            for idx_ttm, ttm in enumerate(ttms):
                forward0 = futoption_chain_fit2.forwards[idx_ttm]
                strikes_ttm = strikes_ttms_mc[idx_ttm]
                normal_vols_ttm = futoption_chain_fit2.ivs_call_ttms[idx_ttm]
                optiontypes = np.repeat('P', strikes_ttm.size)
                ttm_ = np.array([ttm])

                _, mc_ivols, mc_ivols_ups, mc_ivols_downs = calc_mc_vols(basis_type=basis_type,
                                                                         params=params_mc,
                                                                         ttm=ttm,
                                                                         lag=0.0,  # TODO: review it for mid-curve options
                                                                         forward=forward0,
                                                                         strikes=strikes_ttm,
                                                                         optiontypes=optiontypes,
                                                                         measure_type=Measure.FORWARD,
                                                                         T_fwd=ttm,
                                                                         nb_path=nb_path,
                                                                         seed=seed)
                model_prices_ttm = List()
                model_ivs_ttm = List()
                for expansion_order in [ExpansionOrder.FIRST, ExpansionOrder.SECOND]:

                    model_prices_ttm0, model_ivs_ttm0 = logsv_chain_de_pricer(params=params0,
                                                                            t_grid=t_grid,
                                                                            ttms=ttm_,
                                                                            forwards=[[forward0]],
                                                                            strikes_ttms=[[strikes_ttm]],
                                                                            optiontypes_ttms=[optiontypes],
                                                                            do_control_variate=False,
                                                                            is_stiff_solver=is_stiff_solvers[idx_ttm],
                                                                            expansion_order=expansion_order,
                                                                            underlying_type=UnderlyingType.FUTURES,
                                                                            lag=0,  # TODO: review it for mid-curve options
                                                                            x0=None,
                                                                            y0=None)
                    model_prices_ttm.append(model_prices_ttm0)
                    model_ivs_ttm.append(model_ivs_ttm0)


                model_ivs_ttms.append(model_ivs_ttm)
                mc_ivols_ttms.append(mc_ivols)
                mc_ivols_ups_ttms.append(mc_ivols_ups)
                mc_ivols_downs_ttms.append(mc_ivols_downs)

            # print(model_ivs_ttms)
            # print(mc_ivols_ttms)
            # print(mc_ivols_ups_ttms)
            # print(mc_ivols_downs_ttms)

            nb_cols = futoption_chain_fit2.ttms.size
            with sns.axes_style('darkgrid'):
                fig, axs = plt.subplots(1, nb_cols, figsize=(8*nb_cols, 6), tight_layout=True)
            colors = ['blue', 'brown']
            for idx, (ttm, model_ivs_ttm) in enumerate(zip(futoption_chain_fit2.ttms, model_ivs_ttms)):
                ax = axs[idx] if nb_cols > 1 else axs
                for oidx, (model_ivs, color) in enumerate(zip(model_ivs_ttm, colors)):
                    model_ivols_pd = pd.Series(model_ivs[0][0], index=strikes_ttms_mc[idx], name=f"AE{oidx+1}")
                    sns.lineplot(data=pd.concat([model_ivols_pd], axis=1), ax=ax, palette=[color])
                df2 = pd.DataFrame(np.array([mc_ivols_ups_ttms[idx][0], mc_ivols_downs_ttms[idx][0]]).T,
                                  index=strikes_ttms_mc[idx], columns=[f"MC+0.95ci", f"MC-0.95ci"])
                df = pd.DataFrame(futoption_chain_fit2.ivs_call_ttms[idx],
                                  index=futoption_chain_fit2.strikes_ttms[idx], columns=[f"Mkt"])
                # sns.scatterplot(data=df, palette=['black'], markers=['D'], ax=ax)
                sns.scatterplot(data=df2, palette=['green', 'red'], markers=[7, 6], ax=ax)
                title = f"{curr}: {futoption_chain_fit2.ttms_ids[idx]} market data, seed = {seed:.0f}, mult = {mult}"
                title = f"{curr}: {futoption_chain_fit2.ttms_ids[idx]} market data"
                ax.set_title(title, color='darkblue')
                ax.set_xticklabels(['{:.0f}'.format(x * 10000, 2) for x in ax.get_xticks()])
                ax.set_yticklabels(['{:.1f}'.format(x * 10000, 2) for x in ax.get_yticks()])

            filename = f"futures_mc_vs_ae1.pdf"
            # fig.savefig(f"..//draft//figures//fhjm//{filename}")
    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.CALIBRATE_LOGSV_FUT

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
