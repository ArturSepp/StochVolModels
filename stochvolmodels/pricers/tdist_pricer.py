"""
implementation of gaussian mixture pricer and calibration
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import minimize
from numba.typed import List
from typing import Tuple
from enum import Enum

# sv 
import stochvolmodels.pricers.analytic.tdist as td
from stochvolmodels.utils.funcs import to_flat_np_array, timer
from stochvolmodels.pricers.model_pricer import ModelParams, ModelPricer
from stochvolmodels.utils.config import VariableType

# data
from stochvolmodels.data.option_chain import OptionChain


@dataclass
class TdistParams(ModelParams):
    drift: float
    vol: float
    nu: float
    ttm: float  # ttm is important as all params are fixed to this ttm, it is not part of calibration


class TdistPricer(ModelPricer):

    def price_chain(self, option_chain: OptionChain, params: TdistParams, **kwargs) -> np.ndarray:
        """
        implementation of generic method price_chain using heston wrapper for tdist prices
        """
        model_prices_ttms = tdist_vanilla_chain_pricer(drift=params.drift,
                                                       vol=params.vol,
                                                       nu=params.nu,
                                                       ttms=option_chain.ttms,
                                                       forwards=option_chain.forwards,
                                                       strikes_ttms=option_chain.strikes_ttms,
                                                       optiontypes_ttms=option_chain.optiontypes_ttms,
                                                       discfactors=option_chain.discfactors)

        return model_prices_ttms

    def model_mc_price_chain(self, option_chain: OptionChain, params: TdistParams,
                             nb_path: int = 100000,
                             variable_type: VariableType = VariableType.LOG_RETURN,
                             **kwargs
                             ) -> (List[np.ndarray], List[np.ndarray]):
        raise NotImplementedError

    @timer
    def calibrate_model_params_to_chain_slice(self,
                                              option_chain: OptionChain,
                                              params0: TdistParams = None,
                                              is_vega_weighted: bool = True,
                                              is_unit_ttm_vega: bool = False,
                                              **kwargs
                                              ) -> TdistParams:
        """
        implementation of model calibration interface
        fit: TdistParams
        nb: always use option_chain with one slice because we need martingale condition per slice
        """
        ttms = option_chain.ttms
        if len(ttms) > 1:
            raise NotImplementedError(f"cannot calibrate to multiple slices")
        ttm = ttms[0]
        rf_rate = option_chain.discount_rates[0]

        # p0 = (gmm_weights, gmm_mus, gmm_vols)
        if params0 is not None:
            p0 = np.array([params0.vol, params0.nu])
        else:
            p0 = np.array([0.2, 3.0])

        vol_bounds = [(0.05, 10.0)]
        nu_bounds = [(2.01, 20.0)]
        bounds = np.concatenate((vol_bounds, nu_bounds))

        x, y = option_chain.get_chain_data_as_xy()
        market_vols = to_flat_np_array(y)  # market mid quotes
        if is_vega_weighted:
            vegas_ttms = option_chain.get_chain_vegas(is_unit_ttm_vega=is_unit_ttm_vega)
            vegas_ttms = [vegas_ttm/sum(vegas_ttm) for vegas_ttm in vegas_ttms]
            weights = to_flat_np_array(vegas_ttms)
        else:
            weights = np.ones_like(market_vols)

        def parse_model_params(pars: np.ndarray) -> TdistParams:
            vol = pars[0]
            nu = pars[1]
            drift = td.imply_drift_tdist(rf_rate=rf_rate, vol=vol, nu=nu, ttm=ttm)
            return TdistParams(vol=vol, nu=nu, drift=drift, ttm=ttm)

        def objective(pars: np.ndarray, args: np.ndarray) -> float:
            params = parse_model_params(pars=pars)
            model_vols = self.compute_model_ivols_for_chain(option_chain=option_chain, params=params)
            resid = np.nansum(weights * np.square(to_flat_np_array(model_vols) - market_vols))
            return resid

        options = {'disp': True, 'ftol': 1e-10, 'maxiter': 500}
        res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, options=options)
        fit_params = parse_model_params(pars=res.x)

        return fit_params

    @timer
    def calibrate_model_params_to_chain(self,
                                        option_chain: OptionChain,
                                        is_vega_weighted: bool = True,
                                        is_unit_ttm_vega: bool = False,
                                        **kwargs
                                        ) -> List[str, TdistParams]:
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
                                                                 **kwargs)
            fit_params[ids_] = params0
        return fit_params


def tdist_vanilla_chain_pricer(vol: float,
                               nu: float,
                               drift: float,
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
        option_prices_ttm = td.compute_vanilla_price_tdist(spot=forward*discfactor,
                                                           strikes=strikes_ttm,
                                                           ttm=ttm,
                                                           vol=vol,
                                                           nu=nu,
                                                           optiontypes=optiontypes_ttm,
                                                           rf_rate=drift,
                                                           is_compute_risk_neutral_mu=False  # drift is already adjusted
                                                           )
        model_prices_ttms.append(option_prices_ttm)

    return model_prices_ttms


class UnitTests(Enum):
    CALIBRATOR = 1


def run_unit_test(unit_test: UnitTests):

    import seaborn as sns
    from stochvolmodels.utils import plots as plot
    import stochvolmodels.data.test_option_chain as chains

    if unit_test == UnitTests.CALIBRATOR:
        # option_chain = chains.get_btc_test_chain_data()
        option_chain = chains.get_spy_test_chain_data()
        # option_chain = chains.get_gld_test_chain_data()

        tdist_pricer = TdistPricer()
        fit_params = tdist_pricer.calibrate_model_params_to_chain(option_chain=option_chain)

        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(2, 2, figsize=(14, 12), tight_layout=True)
            axs = plot.to_flat_list(axs)

        for idx, (key, params) in enumerate(fit_params.items()):
            print(f"{key}: {params}")
            option_chain0 = OptionChain.get_slices_as_chain(option_chain, ids=[key])
            tdist_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain0, params=params, axs=[axs[idx]])

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.CALIBRATOR

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
