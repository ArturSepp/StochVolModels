"""
implementation of gaussian mixture pricer
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import minimize
from numba import njit
from numba.typed import List
from typing import Tuple
from enum import Enum

from stochvolmodels.utils.funcs import to_flat_np_array, timer, npdf1
import stochvolmodels.pricers.analytic.bsm as bsm
from stochvolmodels.pricers.model_pricer import ModelParams, ModelPricer
from stochvolmodels.utils.config import VariableType

# data
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.data.test_option_chain import get_btc_test_chain_data


@dataclass
class GmmParams(ModelParams):
    gmm_weights: np.ndarray
    gmm_mus: np.ndarray
    gmm_vols: np.ndarray
    ttm: float  # ttm is important as all params are fixed to this ttm, it is not part of calibration

    def sort_by_mus(self):
        indices = np.argsort(self.gmm_mus)
        self.gmm_weights = self.gmm_weights[indices]
        self.gmm_mus = self.gmm_mus[indices]
        self.gmm_vols = self.gmm_vols[indices]

    def get_get_avg_vol(self) -> float:
        return np.sqrt(np.sum(self.gmm_weights*np.square(self.gmm_vols)))

    def compute_state_pdfs(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        state_pdfs = np.zeros((len(x), len(self.gmm_weights)))
        agg_pdf = np.zeros_like(x)
        for idx, (gmm_weight, mu, vol) in enumerate(zip(self.gmm_weights, self.gmm_mus, self.gmm_vols)):
            state_pdf = npdf1(x, mu=mu*self.ttm, vol=vol*np.sqrt(self.ttm))
            state_pdfs[:, idx] = state_pdf
            agg_pdf += gmm_weight*state_pdf
        return state_pdfs, agg_pdf

    def compute_pdf(self, x: np.ndarray):
        pdfs = np.zeros_like(x)
        for gmm_weight, mu, vol in zip(self.gmm_weights, self.gmm_mus, self.gmm_vols):
            pdfs = pdfs + gmm_weight*npdf1(x, mu=mu*self.ttm, vol=vol*np.sqrt(self.ttm))
        return pdfs


class GmmPricer(ModelPricer):

    def price_chain(self, option_chain: OptionChain, params: GmmParams, **kwargs) -> np.ndarray:
        """
        implementation of generic method price_chain using heston wrapper for heston chain
        """
        model_prices_ttms = gmm_vanilla_chain_pricer(gmm_weights=params.gmm_weights,
                                                     gmm_mus=params.gmm_mus,
                                                     gmm_vols=params.gmm_vols,
                                                     ttms=option_chain.ttms,
                                                     forwards=option_chain.forwards,
                                                     strikes_ttms=option_chain.strikes_ttms,
                                                     optiontypes_ttms=option_chain.optiontypes_ttms,
                                                     discfactors=option_chain.discfactors)

        return model_prices_ttms

    def model_mc_price_chain(self, option_chain: OptionChain, params: GmmParams,
                             nb_path: int = 100000,
                             variable_type: VariableType = VariableType.LOG_RETURN,
                             **kwargs
                             ) -> (List[np.ndarray], List[np.ndarray]):
        raise NotImplementedError

    @timer
    def calibrate_model_params_to_chain_slice(self,
                                              option_chain: OptionChain,
                                              params0: GmmParams = None,
                                              is_vega_weighted: bool = True,
                                              is_unit_ttm_vega: bool = False,
                                              n_mixtures: int = 4,
                                              **kwargs
                                              ) -> GmmParams:
        """
        implementation of model calibration interface
        fit: GmmParams
        nb: always use option_chain with one slice because we need martingale condition per slice
        """

        ttms = option_chain.ttms
        if len(ttms) > 1:
            raise NotImplementedError(f"cannot calibrate to multiple slices")
        ttm = ttms[0]
        discfactor = option_chain.discfactors[0]

        # p0 = (gmm_weights, gmm_mus, gmm_vols)
        if params0 is not None:
            p0 = np.concatenate((params0.gmm_weights, params0.gmm_mus, params0.gmm_vols))
            n_mixtures = len(params0.gmm_weights)
        else:
            gmm_weights = np.ones(n_mixtures) / n_mixtures
            gmm_mus = np.zeros(n_mixtures)
            gmm_vols = np.linspace(0.2, 1.0, n_mixtures)
            p0 = np.concatenate((gmm_weights, gmm_mus, gmm_vols))

        gmm_weights_bounds = [(0.0, 1.0)]*n_mixtures
        gmm_mus_bounds = [(-10.0, 10.0)]*n_mixtures
        gmm_vols_bounds = [(0.01, 4.0)]*n_mixtures
        bounds = np.concatenate((gmm_weights_bounds, gmm_mus_bounds, gmm_vols_bounds))

        x, y = option_chain.get_chain_data_as_xy()
        market_vols = to_flat_np_array(y)  # market mid quotes
        if is_vega_weighted:
            vegas_ttms = option_chain.get_chain_vegas(is_unit_ttm_vega=is_unit_ttm_vega)
            vegas_ttms = [vegas_ttm/sum(vegas_ttm) for vegas_ttm in vegas_ttms]
            weights = to_flat_np_array(vegas_ttms)
        else:
            weights = np.ones_like(market_vols)

        def parse_model_params(pars: np.ndarray) -> GmmParams:
            gmm_weights = pars[:n_mixtures]
            gmm_mus = pars[n_mixtures:2*n_mixtures]
            gmm_vols = pars[2*n_mixtures:]
            return GmmParams(gmm_weights=gmm_weights, gmm_mus=gmm_mus, gmm_vols=gmm_vols, ttm=ttm)

        def objective(pars: np.ndarray, args: np.ndarray) -> float:
            params = parse_model_params(pars=pars)
            model_vols = self.compute_model_ivols_for_chain(option_chain=option_chain, params=params)
            resid = np.nansum(weights * np.square(to_flat_np_array(model_vols) - market_vols))
            return resid

        def weights_sum(pars: np.ndarray) -> float:
            params = parse_model_params(pars=pars)
            return np.sum(params.gmm_weights) - 1.0

        def martingale(pars: np.ndarray) -> float:
            params = parse_model_params(pars=pars)
            return np.sum(params.gmm_weights*np.exp((params.gmm_mus+0.5*params.gmm_vols*params.gmm_vols)*ttm)) - discfactor

        constraints = ({'type': 'eq', 'fun': weights_sum}, {'type': 'eq', 'fun': martingale})
        options = {'disp': True, 'ftol': 1e-10, 'maxiter': 300}

        res = minimize(objective, p0, args=None, method='SLSQP', constraints=constraints, bounds=bounds, options=options)
        fit_params = parse_model_params(pars=res.x)
        fit_params.sort_by_mus()

        return fit_params

    @timer
    def calibrate_model_params_to_chain(self,
                                        option_chain: OptionChain,
                                        is_vega_weighted: bool = True,
                                        is_unit_ttm_vega: bool = False,
                                        n_mixtures: int = 4,
                                        **kwargs
                                        ) -> List[str, GmmParams]:
        """
        model params are fitted per slice
        need to splic chain to slices
        """
        fit_params = {}
        params0 = None
        for ids_ in option_chain.ids:
            option_chain0 = OptionChain.get_slices_as_chain(option_chain, ids=[ids_])
            params0 = self.calibrate_model_params_to_chain_slice(option_chain=option_chain0,
                                                                 params0=params0,
                                                                 is_vega_weighted=is_vega_weighted,
                                                                 is_unit_ttm_vega=is_unit_ttm_vega,
                                                                 n_mixtures=n_mixtures,
                                                                 **kwargs)
            fit_params[ids_] = params0
        return fit_params

@njit
def compute_gmm_vanilla_price(gmm_weights: np.ndarray,
                              gmm_mus: np.ndarray,
                              gmm_vols: np.ndarray,
                              ttm: float,
                              forward: float,
                              strike: float,
                              optiontype: str,
                              discfactor: float = 1.0
                              ) -> float:
    """
    bsm deltas for strikes and vols
    """
    price = 0.0
    for gmm_weight, gmm_mu, gmm_vol in zip(gmm_weights, gmm_mus, gmm_vols):
        forward_i = forward*np.exp((gmm_mu+0.5*gmm_vol*gmm_vol)*ttm)
        price_i = bsm.compute_bsm_vanilla_price(forward=forward_i,
                                                strike=strike,
                                                ttm=ttm,
                                                vol=gmm_vol,
                                                optiontype=optiontype,
                                                discfactor=1.0)
        price += gmm_weight * price_i
    return discfactor*price


@njit
def compute_gmm_vanilla_slice_prices(gmm_weights: np.ndarray,
                                     gmm_mus: np.ndarray,
                                     gmm_vols: np.ndarray,
                                     ttm: float,
                                     forward: float,
                                     strikes: np.ndarray,
                                     optiontypes: np.ndarray,
                                     discfactor: float = 1.0
                                     ) -> np.ndarray:
    """
    vectorised bsm deltas for array of aligned strikes, vols, and optiontypes
    """
    def f(strike: float, optiontype: str) -> float:
        return compute_gmm_vanilla_price(gmm_weights=gmm_weights,
                                         gmm_mus=gmm_mus,
                                         gmm_vols=gmm_vols,
                                         forward=forward,
                                         ttm=ttm,
                                         strike=strike,
                                         optiontype=optiontype,
                                         discfactor=discfactor)

    gmm_prices = np.zeros_like(strikes)
    for idx, (strike, optiontype) in enumerate(zip(strikes, optiontypes)):
        gmm_prices[idx] = f(strike, optiontype)
    return gmm_prices


@njit
def gmm_vanilla_chain_pricer(gmm_weights: np.ndarray,
                             gmm_mus: np.ndarray,
                             gmm_vols: np.ndarray,
                             ttms: np.ndarray,
                             forwards: np.ndarray,
                             strikes_ttms: Tuple[np.ndarray, ...],
                             optiontypes_ttms: Tuple[np.ndarray, ...],
                             discfactors: np.ndarray,
                             ) -> np.ndarray:
    """
    vectorised bsm deltas for array of aligned strikes, vols, and optiontypes
    """
    # outputs as numpy lists
    model_prices_ttms = List()
    for ttm, forward, discfactor, strikes_ttm, optiontypes_ttm in zip(ttms, forwards, discfactors, strikes_ttms,
                                                                      optiontypes_ttms):
        option_prices_ttm = compute_gmm_vanilla_slice_prices(gmm_weights=gmm_weights,
                                                             gmm_mus=gmm_mus,
                                                             gmm_vols=gmm_vols,
                                                             ttm=ttm,
                                                             forward=forward,
                                                             strikes=strikes_ttm,
                                                             optiontypes=optiontypes_ttm,
                                                             discfactor=discfactor)
        model_prices_ttms.append(option_prices_ttm)

    return model_prices_ttms


class UnitTests(Enum):
    CALIBRATOR = 1


def run_unit_test(unit_test: UnitTests):

    import seaborn as sns
    import qis as qis

    if unit_test == UnitTests.CALIBRATOR:
        option_chain = get_btc_test_chain_data()
        gmm_pricer = GmmPricer()
        fit_params = gmm_pricer.calibrate_model_params_to_chain(option_chain=option_chain)

        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(2, 2, figsize=(14, 12), tight_layout=True)
            axs = qis.to_flat_list(axs)

        for idx, (key, params) in enumerate(fit_params.items()):
            print(f"{key}: {params}")
            option_chain0 = OptionChain.get_slices_as_chain(option_chain, ids=[key])
            gmm_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain0, params=params, axs=[axs[idx]])

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.CALIBRATOR

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
