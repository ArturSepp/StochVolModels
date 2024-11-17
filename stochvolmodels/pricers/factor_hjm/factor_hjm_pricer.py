import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from enum import Enum
from typing import Dict, Tuple, Optional
from numba.typed import List
import matplotlib

import stochvolmodels.pricers.analytic.bachelier as bachel
from stochvolmodels import ExpansionOrder
from stochvolmodels.data.option_chain import SwOptionChain
from stochvolmodels.pricers.factor_hjm.rate_factor_basis import NelsonSiegel
from stochvolmodels.pricers.factor_hjm.rate_logsv_params import MultiFactRateLogSvParams, TermStructure
from stochvolmodels.pricers.factor_hjm.rate_core import generate_ttms_grid, get_default_swap_term_structure
from stochvolmodels.pricers.factor_hjm.rate_logsv_pricer import simulate_logsv_MF, logsv_chain_de_pricer, Measure


def do_mc_simulation(basis_type: str,
                     ccy: str,
                     ttms: np.ndarray,
                     x0: np.ndarray,
                     y0: np.ndarray,
                     I0: np.ndarray,
                     sigma0: np.ndarray,
                     params: MultiFactRateLogSvParams,
                     nb_path: int,
                     seed: int = None,
                     measure_type: Measure = Measure.RISK_NEUTRAL,
                     ts_sw: np.ndarray = None,
                     bxs: np.ndarray = None,
                     year_days: int = 360,
                     T_fwd: float = None,
                     ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    if basis_type != "NELSON-SIEGEL" :
        raise NotImplementedError
    x0s, y0s, I0s, sigma0s = simulate_logsv_MF(ttms=ttms,
                                               x0=x0,
                                               y0=y0,
                                               I0=I0,
                                               sigma0=sigma0,
                                               theta=params.theta,
                                               kappa1=params.kappa1,
                                               kappa2=params.kappa2,
                                               ts=params.ts,
                                               A=params.A,
                                               R=params.R,
                                               C=params.C,
                                               Omega=params.Omega,
                                               betaxs=params.beta.xs,
                                               volvolxs=params.volvol.xs,
                                               basis=params.basis,
                                               measure_type=measure_type,
                                               nb_path=nb_path,
                                               seed=seed,
                                               ccy=ccy,
                                               ts_sw=ts_sw,
                                               T_fwd=T_fwd,
                                               params0=params,
                                               bxs=bxs,
                                               year_days = year_days)

    return x0s, y0s, I0s, sigma0s

def calc_mc_vols(basis_type: str,
                 params: MultiFactRateLogSvParams,
                 ttm: float,
                 tenors: np.ndarray,
                 forwards: List[np.ndarray],
                 strikes_ttms: List[List[np.ndarray]],
                 optiontypes: np.ndarray,
                 is_annuity_measure: bool,
                 nb_path: int,
                 x0: np.ndarray = None,
                 y0: np.ndarray = None,
                 sigma0: np.ndarray = None,
                 I0: np.ndarray = None,
                 seed: int = None,
                 x_in_delta_space: bool = False) -> (List[np.ndarray], List[np.ndarray]):
    # checks
    assert len(strikes_ttms) == len(tenors)
    assert len(strikes_ttms[0]) == 1
    assert len(forwards) == len(tenors)
    for fwd in forwards:
        assert fwd.shape == (1,)

    ttms = np.array([ttm])
    # we simulate under risk-neutral measure only
    assert is_annuity_measure is False
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

    ts_sws = []
    bond0s = []
    ann0s = []
    swap0s = []
    for tenor in tenors:
        ts_sw = get_default_swap_term_structure(expiry=ttm, tenor=tenor)
        ann0 = params.basis.annuity(t=ttm, ts_sw=ts_sw, x=x0, y=y0, ccy=params.ccy, m=0)[0]
        bond0 = params.basis.bond(0, ttm, x=x0, y=y0, ccy=params.ccy, m=0)[0]
        swap0 = params.basis.swap_rate(t=ttm, ts_sw=ts_sw, x=x0, y=y0, ccy=params.ccy)[0][0]
        ts_sws.append(ts_sw)
        bond0s.append(bond0)
        ann0s.append(ann0)
        swap0s.append(swap0)

    x0s, y0s, I0s, _ = do_mc_simulation(basis_type=basis_type,
                                        ccy=params.ccy,
                                        ttms=ttms,
                                        x0=x0,
                                        y0=y0,
                                        I0=I0,
                                        sigma0=sigma0,
                                        params=params,
                                        nb_path=nb_path,
                                        seed=None,
                                        measure_type=Measure.RISK_NEUTRAL)
    x0 = x0s[-1]
    y0 = y0s[-1]
    I0 = I0s[-1]
    mc_vols = List()
    mc_prices = List()
    mc_vols_ups = List()
    mc_vols_downs = List()
    std_factor = 1.96

    for idx_tenor, tenor in enumerate(tenors):
        ts_sw = ts_sws[idx_tenor]
        ann0 = ann0s[idx_tenor]
        bond0 = bond0s[idx_tenor]
        swap0 = swap0s[idx_tenor]
        strikes_ttm = strikes_ttms[idx_tenor][0]
        swap_mc, ann_mc, numer_mc = params.basis.calculate_swap_rate(ttm=ttm, x0=x0, y0=y0, I0=I0, ts_sw=ts_sw,
                                                                     ccy=params.ccy)
        # calculate option payoffs
        payoffsign = np.where(optiontypes == 'P', -1, 1).astype(float)
        option_mean = np.zeros_like(strikes_ttm)
        option_std = np.zeros_like(strikes_ttm)

        for idx, (strike, sign) in enumerate(zip(strikes_ttm, payoffsign)):
            option_mean[idx] = np.nanmean(1. / numer_mc * ann_mc * np.maximum(sign * (swap_mc - strike), 0)) / ann0 / bond0
            option_std[idx] = np.nanstd(1. / numer_mc * ann_mc * np.maximum(sign * (swap_mc - strike), 0)) / ann0 / bond0
            option_std[idx] = option_std[idx] / np.sqrt(nb_path)

        option_up = option_mean + std_factor * option_std
        option_down = np.maximum(option_mean - std_factor * option_std, 0.0)

        mc_ivols_mid = bachel.infer_normal_ivols_from_chain_prices(ttms=ttms,
                                                                   forwards=forwards,
                                                                   discfactors=np.ones_like(ttms),
                                                                   strikes_ttms=[strikes_ttm],
                                                                   optiontypes_ttms=[optiontypes],
                                                                   model_prices_ttms=[option_mean])
        mc_ivols_up = bachel.infer_normal_ivols_from_chain_prices(ttms=ttms,
                                                                  forwards=forwards,
                                                                  discfactors=np.ones_like(ttms),
                                                                  strikes_ttms=[strikes_ttm],
                                                                  optiontypes_ttms=[optiontypes],
                                                                  model_prices_ttms=[option_up])
        mc_ivols_down = bachel.infer_normal_ivols_from_chain_prices(ttms=ttms,
                                                                    forwards=forwards,
                                                                    discfactors=np.ones_like(ttms),
                                                                    strikes_ttms=[strikes_ttm],
                                                                    optiontypes_ttms=[optiontypes],
                                                                    model_prices_ttms=[option_down])

        mc_vols.append(mc_ivols_mid[0])
        mc_vols_ups.append(mc_ivols_up[0])
        mc_vols_downs.append(mc_ivols_down[0])

        mc_prices.append(option_mean)

    return mc_prices, mc_vols, mc_vols_ups, mc_vols_downs

