"""
Pricer for the log-normal beta SV model with quadratic drift.

Implements the ModelPricer interface for the model of Eq. (3.12) in Sepp and
Rakhmonov (2024). Vanilla and inverse options are valued by Fourier inversion of
the affine expansion of the MGF: the capped payoff of Eq. (5.4) under the
money-market account measure and Eq. (5.13) under the inverse measure, assembled
into calls and puts by Eqs. (5.9) and (5.16), and options on quadratic variance by
Eqs. (5.20) and (5.24). The Monte Carlo path implements the discretization of
Corollary 3.5, Eq. (3.59).

Calibration follows Sec. 6.2: kappa1 and kappa2 are estimated beforehand from the
autocorrelation of the volatility process, leaving sigma0, theta, beta and
epsilon to be fitted by minimizing the vega-weighted mean squared error of
Eq. (6.3) against market implied volatilities.

Reference
---------
A. Sepp and P. Rakhmonov (2024), Log-normal Stochastic Volatility Model with
Quadratic Drift, International Journal of Theoretical and Applied Finance 26(8),
2450003. Equation numbers throughout this module refer to that article.
"""
# package
import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List
from typing import Tuple, Optional
from scipy.optimize import minimize
from enum import Enum

# stochvolmodels
from stochvolmodels.utils.config import VariableType
import stochvolmodels.utils.mgf_pricer as mgfp
from stochvolmodels.utils.mc_payoffs import compute_mc_vars_payoff
from stochvolmodels.utils.funcs import to_flat_np_array, set_time_grid, timer, compute_histogram_data, set_seed

# stochvolmodels pricers
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
import stochvolmodels.pricers.logsv.affine_expansion as afe
from stochvolmodels.pricers.model_pricer import ModelPricer
from stochvolmodels.pricers.logsv.affine_expansion import ExpansionOrder
from stochvolmodels.pricers.logsv.vol_moments_ode import fit_model_vol_backbone_to_varswaps
from stochvolmodels.pricers.rough_logsv.split_simulation import log_spot_full_combined

# data
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data


class LogsvModelCalibrationType(Enum):
    """
    which model parameters the calibration solves for.

    PARAMS4 is the setup of Sec. 6.2, where kappa1 and kappa2 are estimated
    beforehand from the volatility autocorrelation and held fixed; the article
    reports kappa1 = 2.21 and kappa2 = 2.18 for Bitcoin.
    """
    PARAMS4 = 1  # v0, theta, beta, volvol; kappa1, kappa2 are set externally
    PARAMS5 = 2  # v0, theta, kappa1, beta, volvol
    PARAMS6 = 3  # v0, theta, kappa1, kappa2, beta, volvol
    PARAMS_WITH_VARSWAP_FIT = 4  # beta, volvol; kappa1, kappa2 are set externally; term structure of varswap is fit


class ConstraintsType(Enum):
    """
    parameter constraints imposed during calibration.

    MMA_MARTINGALE imposes kappa2 >= beta, which by Theorem 3.7(1) makes Z_t a
    martingale under the MMA measure and by Theorem 3.6 makes Q and the inverse
    measure equivalent. INVERSE_MARTINGALE imposes the stronger kappa2 >= 2 beta
    of Theorem 3.7(2), needed for R_t to be a martingale under the inverse
    measure; Sec. 6.2 calibrates Bitcoin under this constraint. The MOMENT4
    variants add kappa >= 1.5 vartheta^2, the n = 4 diagonal stability condition
    c(4) - 4 kappa < 0 of the moment generator in Eq. (3.48), which keeps the
    fourth moment of the volatility finite.
    """
    UNCONSTRAINT = 1
    MMA_MARTINGALE = 2  # kappa_2 >= beta
    INVERSE_MARTINGALE = 3  # kappa_2 >= 2.0*beta
    MMA_MARTINGALE_MOMENT4 = 4  # kappa_2 >= beta &
    INVERSE_MARTINGALE_MOMENT4 = 5  # kappa_2 >= 2.0*beta


class CalibrationEngine(Enum):
    """
    how model implied volatilities are produced inside the calibration objective.

    ANALYTIC uses the affine expansion of Sec. 4; MC and ROUGH_MC use simulation
    with randoms fixed across the optimizer's iterations.
    """
    ANALYTIC = 1
    MC = 2
    ROUGH_MC = 3


LOGSV_BTC_PARAMS = LogSvParams(sigma0=0.8376, theta=1.0413, kappa1=3.1844, kappa2=3.058, beta=0.1514, volvol=1.8458)


class LogSVPricer(ModelPricer):
    """
    ModelPricer for the log-normal beta SV model of Eq. (3.12).

    Prices option chains by Fourier inversion of the affine expansion, simulates
    the model by the scheme of Eq. (3.59), and calibrates to market implied
    volatilities under the martingale constraints of Theorem 3.7.
    """

    # @timer
    def price_chain(self,
                    option_chain: OptionChain,
                    params: LogSvParams,
                    is_spot_measure: bool = True,
                    **kwargs
                    ) -> List[np.ndarray]:
        """
        price an option chain, implementing the generic ModelPricer interface.

        Delegates to :func:`logsv_chain_pricer`. With ``is_spot_measure=True`` the
        chain is valued under the MMA measure by Eqs. (5.4) and (5.9); with False,
        under the inverse measure by Eqs. (5.13) and (5.16).
        """
        model_prices = logsv_chain_pricer(params=params,
                                          ttms=option_chain.ttms,
                                          forwards=option_chain.forwards,
                                          discfactors=option_chain.discfactors,
                                          strikes_ttms=option_chain.strikes_ttms,
                                          optiontypes_ttms=option_chain.optiontypes_ttms,
                                          is_spot_measure=is_spot_measure,
                                          **kwargs)
        return model_prices

    @timer
    def model_mc_price_chain(self,
                             option_chain: OptionChain,
                             params: LogSvParams,
                             is_spot_measure: bool = True,
                             variable_type: VariableType = VariableType.LOG_RETURN,
                             nb_path: int = 100000,
                             nb_steps: Optional[int] = None,
                             **kwargs
                             ) -> (List[np.ndarray], List[np.ndarray]):
        """
        price an option chain by Monte Carlo rather than the affine expansion.

        Routes to the rough simulator when ``use_rough_mc`` is passed in kwargs, which
        then also requires ``seed``; otherwise to :func:`logsv_mc_chain_pricer`.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            Model prices and their standard errors, one array per maturity slice.
        """
        vol_backbone_etas = params.get_vol_backbone_etas(ttms=option_chain.ttms)
        if 'use_rough_mc' in kwargs and kwargs['use_rough_mc']:
            assert 'seed' in kwargs
            seed = kwargs['seed']
            Z0, Z1, grid_ttms = get_randoms_for_rough_vol_chain_valuation(ttms=option_chain.ttms, nb_path=nb_path,
                                                                          nb_steps_per_year=nb_steps, seed=seed)
            return rough_logsv_mc_chain_pricer_fixed_randoms(ttms=option_chain.ttms,
                                                             forwards=option_chain.forwards,
                                                             discfactors=option_chain.discfactors,
                                                             strikes_ttms=option_chain.strikes_ttms,
                                                             optiontypes_ttms=option_chain.optiontypes_ttms,
                                                             Z0=Z0,
                                                             Z1=Z1,
                                                             sigma0=params.sigma0,
                                                             theta=params.theta,
                                                             kappa1=params.kappa1,
                                                             kappa2=params.kappa2,
                                                             beta=params.beta,
                                                             orthog_vol=params.volvol,
                                                             weights=params.weights,
                                                             nodes=params.nodes,
                                                             timegrids=grid_ttms)
        else:
            return logsv_mc_chain_pricer(v0=params.sigma0,
                                         theta=params.theta,
                                         kappa1=params.kappa1,
                                         kappa2=params.kappa2,
                                         beta=params.beta,
                                         volvol=params.volvol,
                                         vol_backbone_etas=vol_backbone_etas,
                                         ttms=option_chain.ttms,
                                         forwards=option_chain.forwards,
                                         discfactors=option_chain.discfactors,
                                         strikes_ttms=option_chain.strikes_ttms,
                                         optiontypes_ttms=option_chain.optiontypes_ttms,
                                         is_spot_measure=is_spot_measure,
                                         variable_type=variable_type,
                                         nb_path=nb_path,
                                         nb_steps_per_year=nb_steps or int(360 * np.max(option_chain.ttms)) + 1)

    def set_vol_scaler(self, option_chain: OptionChain) -> float:
        """
        set the transform grid scaler from the first ATM volatility of the chain.

        Held fixed across calibration iterations so the Phi grid of Sec. 6.1 does
        not move with sigma0 as the optimizer steps.
        """
        atm0 = option_chain.get_chain_atm_vols()[0]
        ttm0 = option_chain.ttms[0]
        return set_vol_scaler(sigma0=atm0, ttm=ttm0)

    @timer
    def calibrate_model_params_to_chain(self,
                                        option_chain: OptionChain,
                                        params0: LogSvParams,
                                        params_min: LogSvParams = LogSvParams(sigma0=0.1, theta=0.1, kappa1=0.25, kappa2=0.25, beta=-3.0, volvol=0.2),
                                        params_max: LogSvParams = LogSvParams(sigma0=1.5, theta=1.5, kappa1=10.0, kappa2=10.0, beta=3.0, volvol=3.0),
                                        is_vega_weighted: bool = True,
                                        is_unit_ttm_vega: bool = False,
                                        model_calibration_type: LogsvModelCalibrationType = LogsvModelCalibrationType.PARAMS5,
                                        constraints_type: ConstraintsType = ConstraintsType.UNCONSTRAINT,
                                        calibration_engine: CalibrationEngine = CalibrationEngine.ANALYTIC,
                                        nb_path: int = 100000,
                                        nb_steps: int = 360,
                                        seed: int = 10,
                                        **kwargs
                                        ) -> LogSvParams:
        """
        calibrate model parameters to a chain of market implied volatilities.

        Minimizes the weighted mean squared error of Eq. (6.3),

            WMSE = sum_n w_n(T, K) (sigma_n^model(T, K) - sigma_n^implied(T, K))^2,

        with w_n set to Black-Scholes vega when ``is_vega_weighted`` is True, under
        the constraints selected by ``constraints_type``. Sec. 6.2 uses SLSQP;
        ``scipy.optimize.minimize`` is called with its default choice for the
        given bounds and constraints here.

        Parameters
        ----------
        option_chain : OptionChain
            Market chain carrying maturities, strikes, option types and mid vols.
        params0 : LogSvParams
            Starting point. Under PARAMS4 its kappa1 and kappa2 are held fixed.
        params_min, params_max : LogSvParams
            Box bounds on the fitted parameters.
        is_vega_weighted : bool, default True
            Weight residuals by vega, per Eq. (6.3).
        is_unit_ttm_vega : bool, default False
            Normalize vegas within each maturity slice before weighting.
        model_calibration_type : LogsvModelCalibrationType, default PARAMS5
            Which parameters are free.
        constraints_type : ConstraintsType, default UNCONSTRAINT
            Martingale and moment constraints of Theorem 3.7.
        calibration_engine : CalibrationEngine, default ANALYTIC
            Analytic expansion or Monte Carlo inside the objective.
        nb_path, nb_steps, seed : int
            Monte Carlo controls, used only for the MC engines. The randoms are
            drawn once and reused across iterations so the objective is smooth.

        Returns
        -------
        LogSvParams
            Fitted parameters. Compare with Eq. (6.4), which reports
            sigma0 = 0.41, theta = 0.38, beta = 0.50, epsilon = 3.06 for Bitcoin
            on 20 June 2023.
        """
        vol_scaler = self.set_vol_scaler(option_chain=option_chain)

        x, market_vols = option_chain.get_chain_data_as_xy()
        market_vols = to_flat_np_array(market_vols)  # market mid quotes

        if is_vega_weighted:
            vegas_ttms = option_chain.get_chain_vegas(is_unit_ttm_vega=is_unit_ttm_vega)
            # if is_unit_ttm_vega:
            vegas_ttms = [vegas_ttm/sum(vegas_ttm) for vegas_ttm in vegas_ttms]
            weights = to_flat_np_array(vegas_ttms)
        else:
            weights = np.ones_like(market_vols)

        if model_calibration_type == LogsvModelCalibrationType.PARAMS_WITH_VARSWAP_FIT:
            varswap_strikes = option_chain.get_slice_varswap_strikes(floor_with_atm_vols=True)
        else:
            varswap_strikes = None

        def parse_model_params(pars: np.ndarray) -> LogSvParams:
            if model_calibration_type == LogsvModelCalibrationType.PARAMS4:
                fit_params = LogSvParams(sigma0=pars[0],
                                         theta=pars[1],
                                         kappa1=params0.kappa1,
                                         kappa2=params0.kappa2,
                                         beta=pars[2],
                                         volvol=pars[3],
                                         H=params0.H,
                                         nodes=params0.nodes,
                                         weights=params0.weights)
            elif model_calibration_type == LogsvModelCalibrationType.PARAMS5:
                fit_params = LogSvParams(sigma0=pars[0],
                                         theta=pars[1],
                                         kappa1=pars[2],
                                         kappa2=None,
                                         beta=pars[3],
                                         volvol=pars[4],
                                         H=params0.H,
                                         nodes=params0.nodes,
                                         weights=params0.weights)
            elif model_calibration_type == LogsvModelCalibrationType.PARAMS_WITH_VARSWAP_FIT:
                fit_params = LogSvParams(sigma0=params0.sigma0,
                                         theta=params0.theta,
                                         kappa1=params0.kappa1,
                                         kappa2=params0.kappa2,
                                         beta=pars[0],
                                         volvol=pars[1],
                                         H=params0.H,
                                         nodes=params0.nodes,
                                         weights=params0.weights)
                # set model backbone
                vol_backbone = fit_model_vol_backbone_to_varswaps(log_sv_params=fit_params,
                                                                  varswap_strikes=varswap_strikes)
                fit_params.set_vol_backbone(vol_backbone=vol_backbone)

            else:
                raise NotImplementedError(f"{model_calibration_type}")
            return fit_params

        if calibration_engine == CalibrationEngine.MC:
            W0s, W1s, dts = get_randoms_for_chain_valuation(ttms=option_chain.ttms, nb_path=nb_path, nb_steps_per_year=nb_steps, seed=seed)
        if calibration_engine == CalibrationEngine.ROUGH_MC:
            Z0, Z1, grid_ttms = get_randoms_for_rough_vol_chain_valuation(ttms=option_chain.ttms, nb_path=nb_path,
                                                                          nb_steps_per_year=nb_steps, seed=seed)
        def objective(pars: np.ndarray, args: np.ndarray) -> float:
            params = parse_model_params(pars=pars)

            if calibration_engine == CalibrationEngine.ANALYTIC:
                model_vols = self.compute_model_ivols_for_chain(option_chain=option_chain, params=params, vol_scaler=vol_scaler)

            elif calibration_engine == CalibrationEngine.MC:
                option_prices_ttm, option_std_ttm = logsv_mc_chain_pricer_fixed_randoms(ttms=option_chain.ttms,
                                                                                        forwards=option_chain.forwards,
                                                                                        discfactors=option_chain.discfactors,
                                                                                        strikes_ttms=option_chain.strikes_ttms,
                                                                                        optiontypes_ttms=option_chain.optiontypes_ttms,
                                                                                        W0s=W0s,
                                                                                        W1s=W1s,
                                                                                        dts=dts,
                                                                                        v0=params.sigma0,
                                                                                        theta=params.theta,
                                                                                        kappa1=params.kappa1,
                                                                                        kappa2=params.kappa2,
                                                                                        beta=params.beta,
                                                                                        volvol=params.volvol,
                                                                                        vol_backbone_etas=params.get_vol_backbone_etas(ttms=option_chain.ttms))
                model_vols = option_chain.compute_model_ivols_from_chain_data(model_prices=option_prices_ttm)
                # print(f"option_prices_ttm\n{option_prices_ttm}")
                # print(f"model_vols\n{model_vols}")

            elif calibration_engine == CalibrationEngine.ROUGH_MC:
                option_prices_ttm, option_std_ttm = rough_logsv_mc_chain_pricer_fixed_randoms(ttms=option_chain.ttms,
                                                                                              forwards=option_chain.forwards,
                                                                                              discfactors=option_chain.discfactors,
                                                                                              strikes_ttms=option_chain.strikes_ttms,
                                                                                              optiontypes_ttms=option_chain.optiontypes_ttms,
                                                                                              Z0=Z0,
                                                                                              Z1=Z1,
                                                                                              sigma0=params.sigma0,
                                                                                              theta=params.theta,
                                                                                              kappa1=params.kappa1,
                                                                                              kappa2=params.kappa2,
                                                                                              beta=params.beta,
                                                                                              orthog_vol=params.volvol,
                                                                                              weights=params.weights,
                                                                                              nodes=params.nodes,
                                                                                              timegrids=grid_ttms)
                model_vols = option_chain.compute_model_ivols_from_chain_data(model_prices=option_prices_ttm)

            else:
                raise NotImplementedError(f"{calibration_engine}")

            resid = np.nansum(weights * np.square(to_flat_np_array(model_vols) - market_vols))
            return resid

        # parametric constraints
        def martingale_measure(pars: np.ndarray) -> float:
            params = parse_model_params(pars=pars)
            return params.kappa2 - params.beta

        def inverse_measure(pars: np.ndarray) -> float:
            params = parse_model_params(pars=pars)
            return params.kappa2 - 2.0 * params.beta

        def vol_4thmoment_finite(pars: np.ndarray) -> float:
            params = parse_model_params(pars=pars)
            kappa = params.kappa1 + params.kappa2 * params.theta
            return kappa - 1.5 * params.vartheta2

        # set initial params
        if model_calibration_type == LogsvModelCalibrationType.PARAMS4:
            # fit: v0, theta, beta, volvol; kappa1, kappa2 is given with params0
            p0 = np.array([params0.sigma0, params0.theta, params0.beta, params0.volvol])
            bounds = ((params_min.sigma0, params_max.sigma0),
                      (params_min.theta, params_max.theta),
                      (params_min.beta, params_max.beta),
                      (params_min.volvol, params_max.volvol))

        elif model_calibration_type == LogsvModelCalibrationType.PARAMS5:
            # fit: v0, theta, kappa1, beta, volvol; kappa2 is mapped as kappa1 / theta
            p0 = np.array([params0.sigma0, params0.theta, params0.kappa1, params0.beta, params0.volvol])
            bounds = ((params_min.sigma0, params_max.sigma0),
                      (params_min.theta, params_max.theta),
                      (params_min.kappa1, params_max.kappa1),
                      (params_min.beta, params_max.beta),
                      (params_min.volvol, params_max.volvol))

        elif model_calibration_type == LogsvModelCalibrationType.PARAMS_WITH_VARSWAP_FIT:
            # beta, volvol; kappa1, kappa2 are set externally; term structure of varswap is fit
            p0 = np.array([params0.beta, params0.volvol])
            bounds = ((params_min.beta, params_max.beta),
                      (params_min.volvol, params_max.volvol))
        else:
            raise NotImplementedError(f"{model_calibration_type}")

        options = {'disp': True, 'ftol': 1e-8}

        if constraints_type == ConstraintsType.UNCONSTRAINT:
            constraints = None
        elif constraints_type == ConstraintsType.MMA_MARTINGALE:
            constraints = ({'type': 'ineq', 'fun': martingale_measure})
        elif constraints_type == ConstraintsType.INVERSE_MARTINGALE:
            constraints = ({'type': 'ineq', 'fun': inverse_measure})
        elif constraints_type == ConstraintsType.MMA_MARTINGALE_MOMENT4:
            constraints = ({'type': 'ineq', 'fun': martingale_measure}, {'type': 'ineq', 'fun': vol_4thmoment_finite})
        elif constraints_type == ConstraintsType.INVERSE_MARTINGALE_MOMENT4:
            constraints = ({'type': 'ineq', 'fun': inverse_measure}, {'type': 'ineq', 'fun': vol_4thmoment_finite})
        else:
            raise NotImplementedError

        """
        match constraints_type:
            case ConstraintsType.UNCONSTRAINT:
                constraints = None
            case ConstraintsType.MMA_MARTINGALE:
                constraints = ({'type': 'ineq', 'fun': martingale_measure})
            case ConstraintsType.INVERSE_MARTINGALE:
                constraints = ({'type': 'ineq', 'fun': inverse_measure})
            case ConstraintsType.MMA_MARTINGALE_MOMENT4:
                constraints = ({'type': 'ineq', 'fun': martingale_measure}, {'type': 'ineq', 'fun': vol_4thmoment_finite})
            case ConstraintsType.INVERSE_MARTINGALE_MOMENT4:
                constraints = ({'type': 'ineq', 'fun': inverse_measure}, {'type': 'ineq', 'fun': vol_4thmoment_finite})
            case _:
                raise NotImplementedError
        """
        if constraints is not None:
            res = minimize(objective, p0, args=None, method='SLSQP', constraints=constraints, bounds=bounds, options=options)
        else:
            res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, options=options)

        fit_params = parse_model_params(pars=res.x)

        return fit_params

    @timer
    def simulate_vol_paths(self,
                           params: LogSvParams,
                           brownians: np.ndarray = None,
                           ttm: float = 1.0,
                           nb_path: int = 100000,
                           is_spot_measure: bool = True,
                           nb_steps: int = None,
                           year_days: int = 360,
                           **kwargs
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        simulate vols in dt_path grid
        """
        nb_steps = nb_steps or int(np.ceil(year_days * ttm))
        sigma_t, grid_t = simulate_vol_paths(ttm=ttm,
                                             v0=params.sigma0,
                                             theta=params.theta,
                                             kappa1=params.kappa1,
                                             kappa2=params.kappa2,
                                             beta=params.beta,
                                             volvol=params.volvol,
                                             nb_path=nb_path,
                                             is_spot_measure=is_spot_measure,
                                             nb_steps_per_year=nb_steps,
                                             brownians=brownians,
                                             **kwargs)
        return sigma_t, grid_t

    @timer
    def simulate_terminal_values(self,
                                 params: LogSvParams,
                                 ttm: float = 1.0,
                                 nb_path: int = 100000,
                                 is_spot_measure: bool = True,
                                 **kwargs
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        simulate terminal values
        """
        x0, sigma0, qvar0 = simulate_logsv_x_vol_terminal(ttm=ttm,
                                                          x0=np.zeros(nb_path),
                                                          sigma0=params.sigma0 * np.ones(nb_path),
                                                          qvar0=np.zeros(nb_path),
                                                          theta=params.theta,
                                                          kappa1=params.kappa1,
                                                          kappa2=params.kappa2,
                                                          beta=params.beta,
                                                          volvol=params.volvol,
                                                          nb_path=nb_path,
                                                          is_spot_measure=is_spot_measure)
        return x0, sigma0, qvar0

    @timer
    def logsv_pdfs(self,
                   params: LogSvParams,
                   ttm: float,
                   space_grid: np.ndarray,
                   is_stiff_solver: bool = False,
                   is_analytic: bool = False,
                   is_spot_measure: bool = True,
                   expansion_order: ExpansionOrder = ExpansionOrder.SECOND,
                   variable_type: VariableType = VariableType.LOG_RETURN,
                   vol_scaler: float = None
                   ) -> np.ndarray:
        return logsv_pdfs(params=params,
                          ttm=ttm,
                          space_grid=space_grid,
                          is_stiff_solver=is_stiff_solver,
                          is_analytic=is_analytic,
                          is_spot_measure=is_spot_measure,
                          expansion_order=expansion_order,
                          variable_type=variable_type,
                          vol_scaler=vol_scaler
                          )


@njit(cache=False, fastmath=True)
def v0_implied(atm: float, beta: float, volvol: float, theta: float, kappa1: float, ttm: float):
    """
    approximation for short term model atm vol
    """
    beta2 = beta * beta
    volvol2 = volvol*volvol
    vartheta2 = beta2 + volvol2

    def simple():
        return atm - (beta2 + volvol2) * ttm / 4.0

    if np.abs(beta) > 1.0:  # cannot use approximation when beta is too high
        v0 = simple()
    else:
        numer = -24.0 - beta2 * ttm - 2.0 * vartheta2 * ttm + 12.0 * kappa1 * ttm + np.sqrt(np.square(24.0 + beta2 * ttm + 2.0 * vartheta2 * ttm - 12.0 * kappa1 * ttm) - 288.0 * beta * ttm * (
                -2.0 * atm + theta * kappa1 * ttm))
        denumer = 12.0 * beta * ttm
        if np.abs(denumer) > 1e-10:
            v0 = numer / denumer
        else:
            v0 = atm - (np.square(beta) + np.square(volvol)) * ttm / 4.0
    return v0


def set_vol_scaler(sigma0: float, ttm: float) -> float:
    """transform grid scaler from the ATM volatility and the shortest maturity."""
    return sigma0 * np.sqrt(np.minimum(np.min(ttm), 0.5 / 12.0))  # lower bound is two weeks


def logsv_chain_pricer(params: LogSvParams,
                       ttms: np.ndarray,
                       forwards: np.ndarray,
                       discfactors: np.ndarray,
                       strikes_ttms: List[np.ndarray],
                       optiontypes_ttms: List[np.ndarray],
                       is_stiff_solver: bool = False,
                       is_analytic: bool = False,
                       is_spot_measure: bool = True,
                       expansion_order: ExpansionOrder = ExpansionOrder.SECOND,
                       variable_type: VariableType = VariableType.LOG_RETURN,
                       vol_scaler: float = None,
                       **kwargs
                       ) -> List[np.ndarray]:
    """
    wrapper to price option chain on variable_type
    to do: do numba implementation using numba consistent solver
    """
    # starting values
    if vol_scaler is None:  # for calibrations we fix one vol_scaler so the grid is not affected by v0
        vol_scaler = set_vol_scaler(sigma0=params.sigma0, ttm=np.min(ttms))

    phi_grid, psi_grid, theta_grid = mgfp.get_transform_var_grid(variable_type=variable_type,
                                                                 is_spot_measure=is_spot_measure,
                                                                 vol_scaler=vol_scaler)

    a_t0 = np.zeros((phi_grid.shape[0], afe.get_expansion_n(expansion_order)), dtype=np.complex128)
    ttm0 = 0.0

    # outputs as numpy lists
    model_prices_ttms = List()
    for ttm, forward, strikes_ttm, optiontypes_ttm, discfactor in zip(ttms, forwards, strikes_ttms, optiontypes_ttms, discfactors):
        vol_backbone_eta = params.get_vol_backbone_eta(tau=ttm)
        a_t0, log_mgf_grid = afe.compute_logsv_a_mgf_grid(ttm=ttm - ttm0,
                                                          phi_grid=phi_grid,
                                                          psi_grid=psi_grid,
                                                          theta_grid=theta_grid,
                                                          a_t0=a_t0,
                                                          is_analytic=is_analytic,
                                                          expansion_order=expansion_order,
                                                          is_stiff_solver=is_stiff_solver,
                                                          is_spot_measure=is_spot_measure,
                                                          **params.to_dict(),
                                                          vol_backbone_eta=vol_backbone_eta)

        if variable_type == VariableType.LOG_RETURN:
            option_prices = mgfp.vanilla_slice_pricer_with_mgf_grid(log_mgf_grid=log_mgf_grid,
                                                                    phi_grid=phi_grid,
                                                                    forward=forward,
                                                                    strikes=strikes_ttm,
                                                                    optiontypes=optiontypes_ttm,
                                                                    discfactor=discfactor,
                                                                    is_spot_measure=is_spot_measure)

        elif variable_type == VariableType.Q_VAR:
            option_prices = mgfp.slice_qvar_pricer_with_a_grid(log_mgf_grid=log_mgf_grid,
                                                               psi_grid=psi_grid,
                                                               ttm=ttm,
                                                               forward=forward,
                                                               strikes=strikes_ttm,
                                                               optiontypes=optiontypes_ttm,
                                                               discfactor=discfactor,
                                                               is_spot_measure=is_spot_measure)

        else:
            raise NotImplementedError

        model_prices_ttms.append(option_prices)
        ttm0 = ttm

    return model_prices_ttms


def logsv_pdfs(params: LogSvParams,
               ttm: float,
               space_grid: np.ndarray,
               is_stiff_solver: bool = False,
               is_analytic: bool = False,
               is_spot_measure: bool = True,
               expansion_order: ExpansionOrder = ExpansionOrder.SECOND,
               variable_type: VariableType = VariableType.LOG_RETURN,
               vol_scaler: float = None
               ) -> np.ndarray:
    """
    wrapper to compute model pdfs
    """
    # starting values
    if vol_scaler is None:  # for calibrations we fix one vol_scaler so the grid is not affected by v0
        vol_scaler = set_vol_scaler(sigma0=params.sigma0, ttm=ttm)

    phi_grid, psi_grid, theta_grid = mgfp.get_transform_var_grid(variable_type=variable_type,
                                                                 is_spot_measure=is_spot_measure,
                                                                 vol_scaler=vol_scaler)

    a_t0 = afe.get_init_conditions_a(phi_grid=phi_grid,
                                     psi_grid=psi_grid,
                                     theta_grid=theta_grid,
                                     n_terms=afe.get_expansion_n(expansion_order=expansion_order),
                                     variable_type=variable_type)

    # compute mgf
    a_t0, log_mgf_grid = afe.compute_logsv_a_mgf_grid(ttm=ttm,
                                                      phi_grid=phi_grid,
                                                      psi_grid=psi_grid,
                                                      theta_grid=theta_grid,
                                                      a_t0=a_t0,
                                                      is_analytic=is_analytic,
                                                      expansion_order=expansion_order,
                                                      is_stiff_solver=is_stiff_solver,
                                                      is_spot_measure=is_spot_measure,
                                                      **params.to_dict())

    # outputs as numpy lists
    if variable_type == VariableType.LOG_RETURN:
        transform_var_grid = phi_grid
        shift = 0.0
        scale = 1.0
    elif variable_type == VariableType.Q_VAR:  # scaled by ttm
        transform_var_grid = psi_grid
        shift = 0.0
        scale = 1.0 / ttm
    elif variable_type == VariableType.SIGMA:
        transform_var_grid = theta_grid
        shift = params.theta
        scale = 1.0
    else:
        raise NotImplementedError

    pdf = mgfp.pdf_with_mgf_grid(log_mgf_grid=log_mgf_grid,
                                 transform_var_grid=transform_var_grid,
                                 space_grid=space_grid,
                                 shift=shift,
                                 scale=scale)
    pdf = pdf / scale
    return pdf


@njit(cache=False, fastmath=True)
def logsv_mc_chain_pricer(ttms: np.ndarray,
                          forwards: np.ndarray,
                          discfactors: np.ndarray,
                          strikes_ttms: Tuple[np.ndarray,...],
                          optiontypes_ttms: Tuple[np.ndarray, ...],
                          v0: float,
                          theta: float,
                          kappa1: float,
                          kappa2: float,
                          beta: float,
                          volvol: float,
                          vol_backbone_etas: np.ndarray,
                          is_spot_measure: bool = True,
                          nb_path: int = 100000,
                          nb_steps_per_year: int = 360,
                          variable_type: VariableType = VariableType.LOG_RETURN
                          ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    price an option chain by simulating the scheme of Eq. (3.59) slice by slice.

    Each maturity is simulated from the terminal state of the previous one, so the
    chain shares a single path set. Returns model prices and their Monte Carlo
    standard errors, one array per slice.
    """
    # starting values
    x0 = np.zeros(nb_path)
    qvar0 = np.zeros(nb_path)
    sigma0 = v0*np.ones(nb_path)
    ttm0 = 0.0

    # outputs as numpy lists
    option_prices_ttm = List()
    option_std_ttm = List()
    for ttm, forward, discfactor, strikes_ttm, optiontypes_ttm, vol_backbone_eta in zip(ttms, forwards, discfactors,
                                                                                        strikes_ttms, optiontypes_ttms,
                                                                                        vol_backbone_etas):
        x0, sigma0, qvar0 = simulate_logsv_x_vol_terminal(ttm=ttm - ttm0,
                                                          x0=x0,
                                                          sigma0=sigma0,
                                                          qvar0=qvar0,
                                                          theta=theta,
                                                          kappa1=kappa1,
                                                          kappa2=kappa2,
                                                          beta=beta,
                                                          volvol=volvol,
                                                          vol_backbone_eta=vol_backbone_eta,
                                                          nb_path=nb_path,
                                                          nb_steps_per_year=nb_steps_per_year,
                                                          is_spot_measure=is_spot_measure)
        ttm0 = ttm
        option_prices, option_std = compute_mc_vars_payoff(x0=x0, sigma0=sigma0, qvar0=qvar0,
                                                           ttm=ttm,
                                                           forward=forward,
                                                           strikes_ttm=strikes_ttm,
                                                           optiontypes_ttm=optiontypes_ttm,
                                                           discfactor=discfactor,
                                                           variable_type=variable_type)
        option_prices_ttm.append(option_prices)
        option_std_ttm.append(option_std)

    return option_prices_ttm, option_std_ttm


def simulate_vol_paths(ttm: float,
                       v0: float,
                       theta: float,
                       kappa1: float,
                       kappa2: float,
                       beta: float,
                       volvol: float,
                       is_spot_measure: bool = True,
                       nb_path: int = 100000,
                       nb_steps_per_year: int = 360,
                       brownians: np.ndarray = None,
                       **kwargs
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    simulate paths of the volatility process on [0, ttm].

    Steps the log-volatility L_t = ln sigma_t of Eq. (3.55), whose drift is

        zeta(L) = (-kappa1 + kappa2 theta - 0.5 vartheta^2)
                  + kappa1 theta exp(-L) - kappa2 exp(L),

    with ``adj`` shifting the quadratic coefficient from kappa2 to kappa2 - beta
    under the inverse measure, per Eq. (3.26).

    Parameters
    ----------
    ttm : float
        Horizon in years.
    v0 : float
        Initial volatility sigma_0.
    theta, kappa1, kappa2, beta, volvol : float
        Model parameters of Eq. (3.12).
    is_spot_measure : bool, default True
        MMA measure when True, inverse measure when False.
    nb_path : int, default 100000
        Number of paths.
    nb_steps_per_year : int, default 360
        Steps per year. The article uses daily steps.
    brownians : np.ndarray, default None
        Pre-drawn scaled increments of shape (nb_steps, nb_path). Drawn internally
        when None.

    Returns
    -------
    sigma_t : np.ndarray
        Volatility paths including the initial value in the first row.
    grid_t : np.ndarray
        Time grid.

    Notes
    -----
    The drift is evaluated at the current step, so this is an explicit Euler
    scheme on the Lamperti transform, not the backward Euler-Maruyama scheme of
    Eq. (3.56) for which Theorem 3.8 proves strong order 1.
    """
    sigma0 = v0 * np.ones(nb_path)

    nb_steps, dt, grid_t = set_time_grid(ttm=ttm, nb_steps_per_year=nb_steps_per_year)

    if brownians is None:
        brownians = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))

    if is_spot_measure:
        alpha, adj = -1.0, 0.0
    else:
        alpha, adj = 1.0, beta

    vartheta2 = beta*beta + volvol*volvol
    vartheta = np.sqrt(vartheta2)
    vol_var = np.log(sigma0)
    sigma_t = np.zeros((nb_steps + 1, nb_path))  # one row per step plus the initial value at t0 = 0
    sigma_t[0, :] = sigma0  # keep first value
    for t_, w1_ in enumerate(brownians):
        vol_var = vol_var + ((kappa1 * theta / sigma0 - kappa1) + kappa2*(theta-sigma0) + adj*sigma0 - 0.5*vartheta2) * dt + vartheta*w1_
        sigma0 = np.exp(vol_var)
        sigma_t[t_+1, :] = sigma0

    return sigma_t, grid_t


@njit(cache=False, fastmath=False)
def simulate_logsv_x_vol_terminal(ttm: float,
                                  x0:  np.ndarray,
                                  sigma0: np.ndarray,
                                  qvar0: np.ndarray,
                                  theta: float,
                                  kappa1: float,
                                  kappa2: float,
                                  beta: float,
                                  volvol: float,
                                  vol_backbone_eta: float = 1.0,
                                  is_spot_measure: bool = True,
                                  nb_path: int = 100000,
                                  nb_steps_per_year: int = 360,
                                  W0: Optional[np.ndarray] = None,
                                  W1: Optional[np.ndarray] = None,
                                  dt: Optional[float] = None
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    simulate terminal log-return, volatility and quadratic variance.

    Implements the discretization of Corollary 3.5, Eq. (3.59): the log-price
    X_t of Eq. (3.16), the log-volatility L_t of Eq. (3.55) and the quadratic
    variance I_t of Eq. (3.12), driven by two independent Brownian motions
    W^(0) and W^(1). Under the inverse measure the sign of the X drift flips, per
    the dynamics of Eq. (3.36).

    Parameters
    ----------
    ttm : float
        Horizon in years.
    x0, sigma0, qvar0 : np.ndarray
        Initial values, either scalars broadcast to nb_path or full path vectors,
        so that a chain can be simulated maturity by maturity from the previous
        terminal state.
    theta, kappa1, kappa2, beta, volvol : float
        Model parameters of Eq. (3.12).
    vol_backbone_eta : float, default 1.0
        Maturity scaling of theta. 1.0 reproduces the article.
    is_spot_measure : bool, default True
        MMA measure when True, inverse measure when False.
    nb_path : int, default 100000
        Number of paths. Figs. 6 and 9 use 400,000.
    nb_steps_per_year : int, default 360
        Steps per year.
    W0, W1 : Optional[np.ndarray], default None
        Pre-drawn scaled Brownian increments. Supplying them fixes the randoms
        across calibration iterations.
    dt : Optional[float], default None
        Step size matching W0 and W1 when those are supplied.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Terminal log-return, volatility and quadratic variance, each of length
        nb_path.
    """
    if x0.shape[0] == 1:  # initial value
        x0 = x0*np.zeros(nb_path)
    else:
        assert x0.shape[0] == nb_path

    if qvar0.shape[0] == 1:  # initial value
        qvar0 = np.zeros(nb_path)
    else:
        assert qvar0.shape[0] == nb_path

    if sigma0.shape[0] == 1:
        sigma0 = sigma0 * np.ones(nb_path)
    else:
        assert sigma0.shape[0] == nb_path
    if W0 is None and W1 is None:
        nb_steps1, dt, grid_t = set_time_grid(ttm=ttm, nb_steps_per_year=nb_steps_per_year)
        # print(f"nb_steps1={nb_steps1}, dt={dt}")
        sdt = np.sqrt(dt)
        W0_ = sdt * np.random.normal(0, 1, size=(nb_steps1, nb_path))
        W1_ = sdt * np.random.normal(0, 1, size=(nb_steps1, nb_path))
    else:
        sdt = np.sqrt(dt)
        W0_ = sdt * W0
        W1_ = sdt * W1

    if is_spot_measure:
        alpha, adj = -1.0, 0.0
    else:
        alpha, adj = 1.0, beta*vol_backbone_eta  # ? vol_backbone_eta

    vartheta2 = beta*beta + volvol*volvol
    vol_backbone_eta2 = vol_backbone_eta * vol_backbone_eta
    vol_var = np.log(sigma0)
    for t_, (w0, w1) in enumerate(zip(W0_, W1_)):
        sigma0_2dt = vol_backbone_eta2 * sigma0 * sigma0 * dt
        x0 = x0 + alpha * 0.5 * sigma0_2dt + vol_backbone_eta * sigma0 * w0
        vol_var = vol_var + ((kappa1 * theta / sigma0 - kappa1) + kappa2*(theta-sigma0) + adj*sigma0 - 0.5*vartheta2) * dt + beta*w0+volvol*w1
        sigma0 = np.exp(vol_var)
        qvar0 = qvar0 + 0.5*(sigma0_2dt + vol_backbone_eta2 * sigma0 * sigma0 * dt)

    return x0, sigma0, qvar0



def get_randoms_for_chain_valuation(ttms: np.ndarray,
                                    nb_path: int = 100000,
                                    nb_steps_per_year: int = 360,
                                    seed: int = 10
                                    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    we need to fix random normals for subsequent evaluation using mc slices
    outputs as numpy lists
    """
    #
    set_seed(seed)
    W0s = List()
    W1s = List()
    dts = List()
    ttm0 = 0.0
    for ttm in ttms:
        # qqq
        nb_steps_, dt, grid_t = set_time_grid(ttm=ttm - ttm0, nb_steps_per_year=nb_steps_per_year)
        W0s.append(np.random.normal(0, 1, size=(nb_steps_, nb_path)))
        W1s.append(np.random.normal(0, 1, size=(nb_steps_, nb_path)))
        dts.append(dt)
        ttm0 = ttm
    return W0s, W1s, dts

def get_randoms_for_rough_vol_chain_valuation(ttms: np.ndarray,
                                    nb_path: int = 100000,
                                    nb_steps_per_year: int = 360,
                                    seed: int = 10
                                    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    draw the Brownian increments for the rough-kernel chain valuation.

    Returns the two increment arrays and the per-maturity time grids, so that the
    same randoms are reused across calibration iterations.
    """
    np.random.seed(seed)
    grid_ttms = List()
    nb_steps_ttms = np.zeros_like(ttms).astype(int)
    for i, ttm in enumerate(ttms):
        nb_steps, dt, grid_t = set_time_grid(ttm, nb_steps_per_year)
        nb_steps_ttms[i] = nb_steps
        grid_ttms.append(grid_t)
    Z0 = np.random.normal(0, 1, size=(nb_steps_ttms[-1], nb_path))
    Z1 = np.random.normal(0, 1, size=(nb_steps_ttms[-1], nb_path))

    return Z0, Z1, grid_ttms


@njit(cache=False, fastmath=True)
def logsv_mc_chain_pricer_fixed_randoms(ttms: np.ndarray,
                                        forwards: np.ndarray,
                                        discfactors: np.ndarray,
                                        strikes_ttms: Tuple[np.ndarray,...],
                                        optiontypes_ttms: Tuple[np.ndarray, ...],
                                        W0s: Tuple[np.ndarray, ...],
                                        W1s: Tuple[np.ndarray, ...],
                                        dts: Tuple[np.ndarray, ...],
                                        v0: float,
                                        theta: float,
                                        kappa1: float,
                                        kappa2: float,
                                        beta: float,
                                        volvol: float,
                                        vol_backbone_etas: np.ndarray,
                                        is_spot_measure: bool = True,
                                        variable_type: VariableType = VariableType.LOG_RETURN
                                        ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    chain valuation using fixed randoms
    """
    # starting values
    nb_path = W0s[0].shape[1]
    x0 = np.zeros(nb_path)
    qvar0 = np.zeros(nb_path)
    sigma0 = v0*np.ones(nb_path)
    ttm0 = 0.0

    # outputs as numpy lists
    option_prices_ttm = List()
    option_std_ttm = List()
    for ttm, forward, discfactor, strikes_ttm, optiontypes_ttm, vol_backbone_eta, W0, W1, dt in zip(ttms, forwards, discfactors,
                                                                                                strikes_ttms, optiontypes_ttms,
                                                                                                vol_backbone_etas,
                                                                                                W0s, W1s, dts):
        x0, sigma0, qvar0 = simulate_logsv_x_vol_terminal(ttm=ttm - ttm0,
                                                          x0=x0,
                                                          sigma0=sigma0,
                                                          qvar0=qvar0,
                                                          theta=theta,
                                                          kappa1=kappa1,
                                                          kappa2=kappa2,
                                                          beta=beta,
                                                          volvol=volvol,
                                                          vol_backbone_eta=vol_backbone_eta,
                                                          nb_path=nb_path,
                                                          dt=dt,
                                                          is_spot_measure=is_spot_measure,
                                                          W0=W0,
                                                          W1=W1)
        ttm0 = ttm
        option_prices, option_std = compute_mc_vars_payoff(x0=x0, sigma0=sigma0, qvar0=qvar0,
                                                           ttm=ttm,
                                                           forward=forward,
                                                           strikes_ttm=strikes_ttm,
                                                           optiontypes_ttm=optiontypes_ttm,
                                                           discfactor=discfactor,
                                                           variable_type=variable_type)
        option_prices_ttm.append(option_prices)
        option_std_ttm.append(option_std)

    return option_prices_ttm, option_std_ttm

def rough_logsv_mc_chain_pricer_fixed_randoms(ttms: np.ndarray,
                                              forwards: np.ndarray,
                                              discfactors: np.ndarray,
                                              strikes_ttms: Tuple[np.ndarray, ...],
                                              optiontypes_ttms: Tuple[np.ndarray, ...],
                                              Z0: np.ndarray,
                                              Z1: np.ndarray,
                                              sigma0: float,
                                              theta: float,
                                              kappa1: float,
                                              kappa2: float,
                                              beta: float,
                                              orthog_vol: float,
                                              weights: np.ndarray,
                                              nodes: np.ndarray,
                                              timegrids: List[np.ndarray],
                                              variable_type: VariableType = VariableType.LOG_RETURN,
                                              debug: bool = False  # print per-slice path diagnostics
                                              ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    assert weights.shape == nodes.shape and weights.ndim == 1
    # assert kappa2 == 0.0
    N = nodes.size
    dtype = weights.dtype
    v0 = np.full((N,), sigma0 / np.sum(weights), dtype=dtype)

    # need to redenote coefficients
    volvol = np.sqrt(beta ** 2 + orthog_vol ** 2)
    rho = beta / volvol

    nb_path = Z0.shape[1]
    v0_vec = np.repeat(v0[:, None], nb_path, axis=1)
    v_init = v0_vec.copy()
    log_s0 = dtype.type(0.0)
    weight_vec = np.repeat(weights[:, None], nb_path, axis=1)
    nodes_vec = np.repeat(nodes[:, None], nb_path, axis=1)

    # outputs as numpy lists
    option_prices_ttm = List()
    option_std_ttm = List()
    for ttm, forward, discfactor, strikes_ttm, optiontypes_ttm, timegrid in zip(ttms, forwards,
                                                                                discfactors,
                                                                                strikes_ttms,
                                                                                optiontypes_ttms,
                                                                                timegrids):
        nb_steps = timegrid.size - 1
        Z0_ = Z0[:nb_steps]
        Z1_ = Z1[:nb_steps]
        log_spot_str, vol_str, qv_str = log_spot_full_combined(nodes_vec, weight_vec, v0_vec, theta, kappa1, kappa2, log_s0,
                                                               v_init, rho, volvol, timegrid, nb_path, Z0_, Z1_)
        if debug:
            print(f"Number of paths with negative vol: {np.sum(weights @ vol_str < 0.0)}, nan vol: {np.count_nonzero(np.isnan(weights @ vol_str))}")
            print(f"Mean spot Strand: {np.mean(np.exp(log_spot_str))}, nan spots: {np.count_nonzero(np.isnan(log_spot_str))}")

        option_prices, option_std = compute_mc_vars_payoff(x0=log_spot_str, sigma0=vol_str, qvar0=qv_str,
                                                           ttm=ttm,
                                                           forward=forward,
                                                           strikes_ttm=strikes_ttm,
                                                           optiontypes_ttm=optiontypes_ttm,
                                                           discfactor=discfactor,
                                                           variable_type=variable_type)
        option_prices_ttm.append(option_prices)
        option_std_ttm.append(option_std)

    return option_prices_ttm, option_std_ttm

class LocalTests(Enum):
    CHAIN_PRICER = 1
    SLICE_PRICER = 2
    CALIBRATOR = 3
    MC_COMPARISION = 4
    MC_COMPARISION_QVAR = 5
    VOL_PATHS = 6
    TERMINAL_VALUES = 7
    MMA_INVERSE_MEASURE_VS_MC = 8


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import stochvolmodels.data.sample_option_chains as chains

    if local_test == LocalTests.CHAIN_PRICER:
        option_chain = get_btc_test_chain_data()
        logsv_pricer = LogSVPricer()
        model_prices = logsv_pricer.price_chain(option_chain=option_chain, params=LOGSV_BTC_PARAMS)
        print(model_prices)
        logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain, params=LOGSV_BTC_PARAMS)

    if local_test == LocalTests.SLICE_PRICER:
        ttm = 1.0
        forward = 1.0
        strikes = np.array([0.9, 1.0, 1.1])
        optiontypes = np.array(['P', 'C', 'C'])

        logsv_pricer = LogSVPricer()
        model_prices, vols = logsv_pricer.price_slice(params=LOGSV_BTC_PARAMS,
                                                      ttm=ttm,
                                                      forward=forward,
                                                      strikes=strikes,
                                                      optiontypes=optiontypes)
        print(model_prices)
        print(vols)

        for strike, optiontype in zip(strikes, optiontypes):
            model_price, vol = logsv_pricer.price_vanilla(params=LOGSV_BTC_PARAMS,
                                                          ttm=ttm,
                                                          forward=forward,
                                                          strike=strike,
                                                          optiontype=optiontype)
            print(f"{model_price}, {vol}")

    elif local_test == LocalTests.CALIBRATOR:
        option_chain = get_btc_test_chain_data()
        logsv_pricer = LogSVPricer()
        fit_params = logsv_pricer.calibrate_model_params_to_chain(option_chain=option_chain,
                                                                  params0=LOGSV_BTC_PARAMS)
        print(fit_params)
        logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain,
                                                 params=fit_params)

    elif local_test == LocalTests.MC_COMPARISION:
        option_chain = get_btc_test_chain_data()
        logsv_pricer = LogSVPricer()
        logsv_pricer.plot_model_ivols_vs_mc(option_chain=option_chain,
                                            params=LOGSV_BTC_PARAMS)

    elif local_test == LocalTests.MC_COMPARISION_QVAR:
        from stochvolmodels.pricers.logsv.vol_moments_ode import compute_analytic_qvar
        logsv_pricer = LogSVPricer()
        ttms = {'1m': 1.0/12.0, '6m': 0.5}
        option_chain = chains.get_qv_options_test_chain_data()
        option_chain = OptionChain.get_slices_as_chain(option_chain, ids=list(ttms.keys()))
        forwards = np.array([compute_analytic_qvar(params=LOGSV_BTC_PARAMS, ttm=ttm, n_terms=4) for ttm in ttms.values()])
        print(f"QV forwards = {forwards}")

        option_chain.forwards = forwards  # replace forwards to imply BSM vols
        option_chain.strikes_ttms = List(forward * strikes_ttm for forward, strikes_ttm in zip(option_chain.forwards, option_chain.strikes_ttms))

        fig = logsv_pricer.plot_model_ivols_vs_mc(option_chain=option_chain,
                                                  params=LOGSV_BTC_PARAMS,
                                                  variable_type=VariableType.Q_VAR)

    elif local_test == LocalTests.VOL_PATHS:
        logsv_pricer = LogSVPricer()
        nb_path = 10
        sigma_t, grid_t = logsv_pricer.simulate_vol_paths(params=LOGSV_BTC_PARAMS,
                                                          nb_path=nb_path,
                                                          nb_steps=360)

        vol_paths = pd.DataFrame(sigma_t, index=grid_t, columns=[f"{x+1}" for x in range(nb_path)])
        print(vol_paths)

    elif local_test == LocalTests.TERMINAL_VALUES:
        logsv_pricer = LogSVPricer()
        params = LOGSV_BTC_PARAMS
        xt, sigmat, qvart = logsv_pricer.simulate_terminal_values(params=params)
        hx = compute_histogram_data(data=xt, x_grid=params.get_x_grid(), name='Log-price')
        hsigmat = compute_histogram_data(data=sigmat, x_grid=params.get_sigma_grid(), name='Sigma')
        hqvar = compute_histogram_data(data=qvart, x_grid=params.get_qvar_grid(), name='Qvar')
        dfs = {'Log-price': hx, 'Sigma': hsigmat, 'Qvar': hqvar}

        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(1, 3, figsize=(18, 10), tight_layout=True)
        for idx, (key, df) in enumerate(dfs.items()):
            axs[idx].fill_between(df.index, np.zeros_like(df.to_numpy()), df.to_numpy(),
                                  facecolor='lightblue', step='mid', alpha=0.8, lw=1.0)
            axs[idx].set_title(key)

    elif local_test == LocalTests.MMA_INVERSE_MEASURE_VS_MC:
        option_chain = get_btc_test_chain_data()
        logsv_pricer = LogSVPricer()
        logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=option_chain,
                                                           params=LOGSV_BTC_PARAMS)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.MC_COMPARISION_QVAR)
