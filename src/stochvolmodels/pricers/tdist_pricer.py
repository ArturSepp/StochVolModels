"""
implementation of gaussian mixture pricer and calibration
"""
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
from numba.typed import List
from typing import Tuple

# sv 
import stochvolmodels.fitters.tdist as td
from stochvolmodels.utils.funcs import to_flat_np_array, timer
from stochvolmodels.pricers.model_pricer import (
    ModelParams,
    ModelPricer,
    validate_optimization_result,
)
from stochvolmodels.utils.config import VariableType

# data
from stochvolmodels.data.option_chain import OptionChain


@dataclass
class TdistParams(ModelParams):
    """
    parameters of the Student-t model: volatility, drift and degrees of freedom nu.

    Terminal log-returns are Student-t with nu > 2, scaled so the variance matches
    vol^2 ttm. Lower nu gives heavier tails and a more pronounced smile.
    """
    drift: float
    vol: float
    nu: float
    ttm: float  # ttm is important as all params are fixed to this ttm, it is not part of calibration


class TdistPricer(ModelPricer):

    """ModelPricer valuing options under a Student-t terminal distribution."""
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
        """price an option chain by Monte Carlo rather than the analytic solution."""
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
            raise NotImplementedError("cannot calibrate to multiple slices")
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
            """map the optimizer parameter vector onto a model parameter object."""
            vol = pars[0]
            nu = pars[1]
            drift = td.imply_drift_tdist(rf_rate=rf_rate, vol=vol, nu=nu, ttm=ttm)
            return TdistParams(vol=vol, nu=nu, drift=drift, ttm=ttm)

        def objective(pars: np.ndarray, args: np.ndarray) -> float:
            """weighted mean squared error between model and market implied volatilities."""
            params = parse_model_params(pars=pars)
            model_vols = self.compute_model_ivols_for_chain(option_chain=option_chain, params=params)
            resid = np.nansum(weights * np.square(to_flat_np_array(model_vols) - market_vols))
            return resid

        options = {'disp': True, 'ftol': 1e-10, 'maxiter': 500}
        res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, options=options)
        fit_params = parse_model_params(
            pars=validate_optimization_result(res, bounds)
        )

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
        discount_rate = -np.log(discfactor) / ttm
        option_prices_ttm = td.compute_vanilla_price_tdist(spot=forward*discfactor,
                                                           strikes=strikes_ttm,
                                                           ttm=ttm,
                                                           vol=vol,
                                                           nu=nu,
                                                           optiontypes=optiontypes_ttm,
                                                           rf_rate=discount_rate,
                                                           risk_neutral_mu=drift,
                                                           )
        model_prices_ttms.append(option_prices_ttm)

    return model_prices_ttms


# Manual scenarios are available in ``stochvolmodels.pricers.tests.tdist_pricer_local``.
