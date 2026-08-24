"""Student-t terminal-distribution pricing and calibration."""
from dataclasses import dataclass
from numbers import Real
from typing import ClassVar, Tuple

import numpy as np
from numba.typed import List
from scipy.optimize import minimize

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
from stochvolmodels.data.option_chain import OptionChain, OptionSlice


@dataclass
class TdistParams(ModelParams):
    """
    parameters of the Student-t model: volatility, drift and degrees of freedom nu.

    The centered arithmetic-return shock is Student-t with nu > 2 and variance
    ``vol**2 * ttm``. The terminal asset is the positive part of spot times one plus
    drift times maturity plus that shock, so the floor creates an atom at zero.
    """
    drift: float
    vol: float
    nu: float
    ttm: float  # Parameters are fixed to this maturity, which is not calibrated.


class TdistPricer(ModelPricer):

    """ModelPricer valuing options under a Student-t terminal distribution."""
    def price_chain(self, option_chain: OptionChain, params: TdistParams, **kwargs) -> np.ndarray:
        """Price all chain slices through the legacy closed-form Student-t kernel."""
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
            model_vols = self.compute_model_ivols_for_chain(
                option_chain=option_chain,
                params=params,
            )
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


@dataclass(frozen=True, init=False)
class TdistTerminalModel:
    """Validated, parameter-bound Student-t law for one-maturity European options.

    The bound drift is consistent with exactly one maturity and discount rate through
    the floored-arithmetic-return martingale equation. Only standard calls and puts are
    supported; the historical ``IC`` and ``IP`` codes do not implement inverse settlement
    and are deliberately rejected at this boundary. Discount consistency uses the
    dimensionless condition ``abs(log(discount * expected_growth)) <= 5e-10``.
    """

    _drift: float
    _vol: float
    _nu: float
    _ttm: float

    _MARTINGALE_LOG_ATOL: ClassVar[float] = 5.0e-10

    def __init__(self, params: TdistParams) -> None:
        """Validate and snapshot one legacy parameter payload."""
        if not isinstance(params, TdistParams):
            raise TypeError("params must be a TdistParams instance")

        values = {
            "drift": params.drift,
            "vol": params.vol,
            "nu": params.nu,
            "ttm": params.ttm,
        }
        for name in ("drift", "vol", "nu", "ttm"):
            value = values[name]
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite real scalar")
        if values["vol"] <= 0.0:
            raise ValueError("vol must be positive")
        if values["nu"] <= 2.0:
            raise ValueError("nu must be greater than 2")
        if values["ttm"] <= 0.0:
            raise ValueError("ttm must be positive")

        for name, value in values.items():
            object.__setattr__(self, f"_{name}", float(value))

    @property
    def params(self) -> TdistParams:
        """Return a detached legacy parameter payload for inspection or facade calls."""
        return TdistParams(
            drift=self._drift,
            vol=self._vol,
            nu=self._nu,
            ttm=self._ttm,
        )

    @property
    def ttm(self) -> float:
        """Return the terminal law's time to maturity in years."""
        return self._ttm

    def _expected_growth(self) -> float:
        """Return expected terminal asset growth before discounting."""
        default_boundary = -(1.0 + self._drift * self._ttm)
        default_probability = td.cdf_tdist(
            x=default_boundary,
            mu=0.0,
            vol=self._vol,
            nu=self._nu,
            ttm=self._ttm,
        )
        truncated_mean = td.cum_mean_tdist(
            x=default_boundary,
            mu=0.0,
            vol=self._vol,
            nu=self._nu,
            ttm=self._ttm,
        )
        expected_growth = (
            (1.0 + self._drift * self._ttm) * (1.0 - default_probability)
            - truncated_mean
        )
        if not np.isfinite(expected_growth) or expected_growth <= 0.0:
            raise ValueError("params imply an invalid expected terminal growth factor")
        return float(expected_growth)

    def _option_chain(self, option_slice: OptionSlice) -> OptionChain:
        """Validate one standard-payoff slice and convert it to the legacy facade input."""
        if not isinstance(option_slice, OptionSlice):
            raise TypeError("option_slice must be an OptionSlice instance")
        if option_slice.ttm != self.ttm:
            raise ValueError("option_slice.ttm must exactly match the bound params.ttm")

        unsupported = set(np.asarray(option_slice.optiontypes).astype(str)) - {"C", "P"}
        if unsupported:
            raise NotImplementedError(
                "TdistTerminalModel supports C/P only; inverse IC/IP settlement is not implemented"
            )

        discounted_growth = option_slice.discfactor * self._expected_growth()
        if not np.isfinite(discounted_growth) or discounted_growth <= 0.0:
            raise ValueError("option_slice discount rate implies invalid discounted growth")
        martingale_log_error = abs(np.log(discounted_growth))
        if martingale_log_error > self._MARTINGALE_LOG_ATOL:
            raise ValueError(
                "option_slice discount rate is inconsistent with the bound drift: "
                f"abs(log(discount * expected_growth)) must not exceed "
                f"{self._MARTINGALE_LOG_ATOL:.0e}"
            )

        return OptionChain.slice_to_chain(
            ttm=option_slice.ttm,
            forward=option_slice.forward,
            strikes=np.asarray(option_slice.strikes, dtype=float),
            optiontypes=np.asarray(option_slice.optiontypes),
            discfactor=option_slice.discfactor,
            id=option_slice.id,
        )

    def price_european(self, option_slice: OptionSlice) -> np.ndarray:
        """Return standard-call/put prices shaped like ``option_slice.strikes``."""
        option_chain = self._option_chain(option_slice)
        prices = TdistPricer().price_chain(option_chain=option_chain, params=self.params)
        return np.asarray(prices[0], dtype=float)

    def implied_vols(self, option_slice: OptionSlice) -> np.ndarray:
        """Return Black implied volatilities shaped like ``option_slice.strikes``."""
        option_chain = self._option_chain(option_slice)
        ivols = TdistPricer().compute_model_ivols_for_chain(
            option_chain=option_chain,
            params=self.params,
        )
        return np.asarray(ivols[0], dtype=float)


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
    for ttm, forward, discfactor, strikes_ttm, optiontypes_ttm in zip(
        ttms, forwards, discfactors, strikes_ttms, optiontypes_ttms
    ):
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


# Manual scenarios are available in ``stochvolmodels.pricers.run_local.tdist_pricer_run``.
