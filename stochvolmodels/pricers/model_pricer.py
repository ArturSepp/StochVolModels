"""
ModelPricer is an interface class for the parent pricing model that defines the following interfaces:
1. price_chain is the abstract method that is implemented in the parent class for a specific model and its params using analytic solution
2. model_mc_price_chain is the interface method for pricing option chain using mc of model dynamics
3. calibrate_model_params_to_chain is the interface method that using model based price_chain
the rest of interface methods are concrete relying on price_chain
market options data is passed using data container ChainData
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba.typed import List
from abc import ABC, abstractmethod
from scipy import stats
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict

from stochvolmodels.pricers.core.config import VariableType
from stochvolmodels.data.option_chain import OptionChain, OptionSlice
from stochvolmodels.utils import plots as plot
from stochvolmodels.utils.funcs import set_seed, update_kwargs

# set global mc seed, for resets call set_seed() locally
set_seed(24)


@dataclass
class ModelParams:
    pass

    @classmethod
    def copy(cls, obj: ModelParams) -> ModelParams:
        return cls(**asdict(obj))


class ModelPricer(ABC):

    def __init__(self):
        super().__init__()

    #########################################################
    #                     generic interfaces
    #########################################################
    @abstractmethod
    def price_chain(self, option_chain: OptionChain, params: ModelParams, **kwargs
                    ) -> List[np.ndarray]:
        """
        abstract method for pricing chain data using model parameters
        recommended as a wrapper for numba implementation
        note that numba.List is equivalent type for Tuple
        """
        pass

    def compute_chain_prices_with_vols(self,
                                       option_chain: OptionChain,
                                       params: ModelParams,
                                       **kwargs
                                       ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        price chain and compute model vols
        """
        model_prices = self.price_chain(option_chain=option_chain, params=params, **kwargs)
        model_ivols = option_chain.compute_model_ivols_from_chain_data(model_prices=model_prices)
        return model_prices, model_ivols

    def compute_model_ivols_for_chain(self,
                                      option_chain: OptionChain,
                                      params: ModelParams,
                                      **kwargs
                                      ) -> List[np.ndarray]:
        """
        price chain and compute model vols
        """
        model_prices, model_ivols = self.compute_chain_prices_with_vols(option_chain=option_chain,
                                                                        params=params,
                                                                        **kwargs)
        return model_ivols

    def model_mc_price_chain(self, option_chain: OptionChain, params: ModelParams, **kwargs
                             ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        abstract method for pricing chain data using simulation of model dynamics
        recommended as a wrapper for numba implementation
        """
        raise NotImplementedError(f"must be implemented in parent class")

    def calibrate_model_params_to_chain(self, option_chain: OptionChain, **kwargs):
        """
        this is core method for model calibration
        we keep as not generic because model implementation may not require calibration of model parameters
        """
        raise NotImplementedError(f"must be implemented in parent class")

    #########################################################
    #          implemented interfaces for pricing
    #########################################################
    def price_slice(self,
                    params: ModelParams,
                    ttm: float,
                    forward: float,
                    strikes: np.ndarray,
                    optiontypes: np.ndarray,
                    discfactor: float = 1.0,
                    **kwargs
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        price single slice_t using model chain pricer
        return price slices and model implied vols
        """
        option_chain = OptionChain.slice_to_chain(ttm=ttm,
                                                  forward=forward,
                                                  strikes=strikes,
                                                  optiontypes=optiontypes,
                                                  discfactor=discfactor)
        model_prices = self.price_chain(option_chain=option_chain,
                                        params=params,
                                        **kwargs)
        model_ivols = option_chain.compute_model_ivols_from_chain_data(model_prices=model_prices)
        return model_prices[0], model_ivols[0]

    def price_vanilla(self,
                      params: ModelParams,
                      ttm: float,
                      forward: float,
                      strike: float,
                      optiontype: str,
                      discfactor: float = 1.0,
                      **kwargs
                      ) -> Tuple[float, float]:
        """
        price single option using slice_t pricer
        return price and model implied vol
        """
        model_prices, model_ivols = self.price_slice(params=params, ttm=ttm, forward=forward,
                                                     strikes=np.array([strike]), optiontypes=np.array([optiontype]),
                                                     discfactor=discfactor, **kwargs)
        return model_prices[0], model_ivols[0]

    #########################################################
    #          implemented interfaces for monte carlo
    #########################################################
    def simulate_vol_paths(self, params: ModelParams, **kwargs) -> (np.ndarray, np.ndarray):
        """
        get grid of vol paths
        """
        raise NotImplementedError(f"must be implemented in parent class")

    def simulate_terminal_values(self, params: ModelParams, **kwargs) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        get realizationss of terminal paths of x, vol, qvar
        """
        raise NotImplementedError(f"must be implemented in parent class")

    #########################################################
    #          implemented interfaces for mc implied vol comptutions
    #########################################################
    def compute_mc_chain_implied_vols(self,
                                      option_chain: OptionChain,
                                      params: ModelParams,
                                      nb_path: int = 100000,
                                      **kwargs
                                      ) -> Tuple[List[np.ndarray], ...]:
        """
        use model_mc_price_chain intefrace to compute model ivols with confidence bound
        """
        model_prices_ttms, option_std_ttms = self.model_mc_price_chain(option_chain=option_chain,
                                                                       params=params,
                                                                       nb_path=nb_path,
                                                                       **kwargs)
        std_factor = 1.96
        model_prices_ttms_ups = List()
        model_prices_ttms_downs = List()
        for model_prices_ttm, option_std_ttm in zip(model_prices_ttms, option_std_ttms):
            model_prices_ttms_ups.append(model_prices_ttm + std_factor * option_std_ttm)
            model_prices_ttms_downs.append(np.maximum(model_prices_ttm - std_factor * option_std_ttm, 1e-10))

        ivols_mid = option_chain.compute_model_ivols_from_chain_data(model_prices=model_prices_ttms)
        ivols_up = option_chain.compute_model_ivols_from_chain_data(model_prices=model_prices_ttms_ups)
        ivols_down = option_chain.compute_model_ivols_from_chain_data(model_prices=model_prices_ttms_downs)
        return model_prices_ttms, model_prices_ttms_ups, model_prices_ttms_downs, ivols_mid, ivols_up, ivols_down, option_std_ttms

    def get_log_return_mc_pdf(self,
                              ttm: float,
                              params: ModelParams,
                              x_grid: np.ndarray,
                              nb_path: int = 100000
                              ) -> np.ndarray:

        t_values = self.simulate_terminal_values(ttm=ttm, params=params, nb_path=nb_path)

        cut_off = 1e16
        # nan can be present too
        inf_nans = np.isnan(t_values)
        inf_pos = np.greater(t_values, cut_off, where=inf_nans == False)
        inf_neg = np.less(t_values, -cut_off, where=inf_nans == False)

        print(f"in mc: num -inf = {np.sum(inf_neg)}, num +inf = {np.sum(inf_pos)}, num nans = {np.sum(inf_nans)}")
        t_values = t_values[np.logical_and(np.logical_and(inf_neg == False, inf_pos == False), inf_nans == False)]
        kernel = stats.gaussian_kde(t_values)
        z = kernel(x_grid)
        # z = np.where(np.greater(z, 0.0), z, 0.0)
        z = z / np.nansum(z)
        return z

    #########################################################
    #          densities
    #########################################################
    def compute_logreturn_pdf(self, params: ModelParams, **kwargs) -> np.ndarray:
        """
        model pdf
        """
        raise NotImplementedError(f"must be implemented in parent class")

    #########################################################
    #          visualization interfaces
    #########################################################
    def plot_model_ivols(self,
                         option_chain: OptionChain,
                         params: ModelParams,
                         is_log_strike_xaxis: bool = False,
                         headers: Optional[List[str]] = None,
                         ax: plt.Subplot = None,
                         **kwargs
                         ) -> Optional[plt.Figure]:
        """
        plot model slice_t vols
        """
        model_ivols = self.compute_model_ivols_for_chain(option_chain=option_chain, params=params, **kwargs)

        if ax is None:
            with sns.axes_style('darkgrid'):
                fig, ax = plt.subplots(1, 1, figsize=plot.FIGSIZE, tight_layout=True)
        else:
            fig = None

        model_vols_ts = []
        for idx, ttm in enumerate(option_chain.ttms):
            if is_log_strike_xaxis:
                strikes = np.log(option_chain.strikes_ttms[idx] / option_chain.forwards[idx])
            else:
                strikes = option_chain.strikes_ttms[idx]

            if option_chain.ids is not None:
                if headers is not None:
                    name = f"{headers[idx]} slice - {option_chain.ids[idx]}"
                else:
                    name = f"Slice - {option_chain.ids[idx]}"
            else:
                name = f"{ttm=:0.2f}"
            model_vols_ts.append(pd.Series(model_ivols[idx], index=strikes, name=name))

        model_vols_ts = pd.concat(model_vols_ts, axis=1)

        plot.model_vols_ts(model_vols=model_vols_ts,
                           title='Model Implied Black Volatilities',
                           xlabel='log-strike' if is_log_strike_xaxis else 'strike',
                           xvar_format='{:0.2f}' if is_log_strike_xaxis else '{:0,.0f}',
                           x_rotation=0,
                           ax=ax,
                           **kwargs)
        return fig

    def plot_model_slices_in_params(self,
                                    option_slice: OptionSlice,
                                    params_dict: Dict[str, ModelParams],
                                    is_log_strike_xaxis: bool = False,
                                    ax: plt.Subplot = None,
                                    **kwargs
                                    ) -> Optional[plt.Figure]:
        """
        plot model slice_t vols
        """
        model_vols_pars = []
        for key, params in params_dict.items():
            prices, model_ivols = self.price_slice(ttm=option_slice.ttm,
                                                   forward=option_slice.forward,
                                                   strikes=option_slice.strikes,
                                                   optiontypes=option_slice.optiontypes,
                                                   params=params,
                                                   **kwargs)
            model_vols_pars.append(pd.Series(model_ivols, index=option_slice.strikes, name=key))

        model_vols_pars = pd.concat(model_vols_pars, axis=1)

        if ax is None:
            with sns.axes_style('darkgrid'):
                fig, ax = plt.subplots(1, 1, figsize=plot.FIGSIZE, tight_layout=True)
        else:
            fig = None

        plot.model_vols_ts(model_vols=model_vols_pars,
                           title='Model Vols',
                           xlabel='log-strike' if is_log_strike_xaxis else 'strike',
                           xvar_format='{:0.2f}' if is_log_strike_xaxis else '{:0,.0f}',
                           x_rotation=0,
                           ax=ax,
                           **kwargs)
        return fig

    def plot_model_ivols_vs_bid_ask(self,
                                    option_chain: OptionChain,
                                    params: ModelParams,
                                    is_log_strike_xaxis: bool = False,
                                    headers: Optional[List[str]] = None,
                                    xvar_format: str = None,
                                    **kwargs
                                    ) -> plt.Figure:
        """
        plot model slice_t vols
        optimized for 2*2 figure
        """
        model_ivols = self.compute_model_ivols_for_chain(option_chain=option_chain, params=params, **kwargs)

        num_slices = len(option_chain.ttms)
        if num_slices == 1:
            with sns.axes_style('darkgrid'):
                fig, ax = plt.subplots(1, 1, figsize=plot.FIGSIZE, tight_layout=True)
            axs = np.array([[ax, np.nan], [np.nan, np.nan]])
        elif num_slices == 2:
            with sns.axes_style('darkgrid'):
                fig, axs = plt.subplots(2, 1, figsize=plot.FIGSIZE, tight_layout=True)
            axs = np.array([[axs[0], np.nan], [axs[1], np.nan]])
        elif num_slices == 4:
            with sns.axes_style('darkgrid'):
                fig, axs = plt.subplots(2, 2, figsize=plot.FIGSIZE, tight_layout=True)
        else:
            raise NotImplementedError

        atm_vols = option_chain.get_chain_atm_vols()
        for idx, ttm in enumerate(option_chain.ttms):
            if is_log_strike_xaxis:
                strikes = np.log(option_chain.strikes_ttms[idx] / option_chain.forwards[idx])
                atm_forward = 0.0
                xvar_format = xvar_format or '{:0.2f}'
                strike_name = 'log-strike'
            else:
                strikes = option_chain.strikes_ttms[idx]
                atm_forward = option_chain.forwards[idx]
                xvar_format = xvar_format or '{:0,.0f}'
                strike_name = 'strike'

            midvols = 0.5 * (option_chain.bid_ivs[idx] + option_chain.ask_ivs[idx])
            mse2 = np.sqrt(np.nanmean(np.power(model_ivols[idx] - midvols, 2)))
            model_vols = pd.Series(model_ivols[idx], index=strikes, name=f"Model Fit: mse={mse2:0.2%}")
            if option_chain.ids is not None:
                if headers is not None:
                    title = f"{headers[idx]} slice - {option_chain.ids[idx]}"
                else:
                    title = f"Slice - {option_chain.ids[idx]}"
            else:
                title = f"{ttm=:0.2f}"

            if is_log_strike_xaxis:
                atm_points = {'ATM': (0.0, atm_vols[idx])}
            else:
                atm_points = {'ATM': (atm_forward, atm_vols[idx])}

            plot.vol_slice_fit(bid_vol=pd.Series(option_chain.bid_ivs[idx], index=strikes),
                               ask_vol=pd.Series(option_chain.ask_ivs[idx], index=strikes),
                               model_vols=model_vols,
                               title=title,
                               atm_points=atm_points,
                               strike_name=strike_name,
                               xvar_format=xvar_format,
                               x_rotation=0,
                               ax=axs[idx % 2][idx // 2],
                               **kwargs)
        return fig

    def plot_model_ivols_vs_mc(self,
                               option_chain: OptionChain,
                               params: ModelParams,
                               is_log_strike_xaxis: bool = False,
                               variable_type: VariableType = VariableType.LOG_RETURN,
                               nb_path: int = 100000,
                               **kwargs
                               ) -> None:
        """
        comparision of model implied vols computed old_analytics vs mc pricer
        optimized for 2*2 figure
        """
        model_ivols = self.compute_model_ivols_for_chain(option_chain=option_chain, params=params, **kwargs)

        mc_prices_ttms, _, _, mc_ivols, mc_ivols_up, mc_ivols_down, _ = self.compute_mc_chain_implied_vols(
            option_chain=option_chain,
            params=params,
            nb_path=nb_path,
            **kwargs)

        with sns.axes_style('darkgrid'):
            if len(option_chain.ttms) > 1:
                fig, axs = plt.subplots(2, len(option_chain.ttms) // 2, figsize=plot.FIGSIZE, tight_layout=True)
            else:
                fig, axs = plt.subplots(1, 1, figsize=plot.FIGSIZE, tight_layout=True)

        for idx, ttm in enumerate(option_chain.ttms):
            if is_log_strike_xaxis:
                strikes = np.log(option_chain.strikes_ttms[idx] / option_chain.forwards[idx])
                xvar_format = '{:0.2f}'
                strike_name = 'log-strike'
            else:
                strikes = option_chain.strikes_ttms[idx]
                if variable_type == VariableType.LOG_RETURN:
                    xvar_format = '{:0,.0f}'
                    strike_name = 'strike'
                else:
                    xvar_format = '{:0.2f}'
                    strike_name = 'QVAR strike'

            mse2 = np.sqrt(np.nanmean(np.power(model_ivols[idx] - mc_ivols[idx], 2)))
            model_vol_t = pd.Series(model_ivols[idx], index=strikes, name=f"Model: mse={mse2:0.2%}")

            if option_chain.ids is not None:
                title = f"{option_chain.ids[idx]}, {ttm=:0.2f}"
            else:
                title = f"{ttm=:0.2f}"

            if len(option_chain.ttms) > 1:
                ax = axs[idx % 2][idx // 2]
            else:
                ax = axs
            plot.vol_slice_fit(bid_vol=pd.Series(mc_ivols_down[idx], index=strikes),
                               ask_vol=pd.Series(mc_ivols_up[idx], index=strikes),
                               model_vols=model_vol_t,
                               title=title,
                               bid_name='MC: -0.95ci',
                               ask_name='MC: +0.95ci',
                               strike_name=strike_name,
                               xvar_format=xvar_format,
                               x_rotation=0,
                               ax=ax,
                               **kwargs)

    def plot_comp_mma_inverse_options_with_mc(self,
                                              option_chain: OptionChain,
                                              params: ModelParams,
                                              variable_type: VariableType = VariableType.LOG_RETURN,
                                              nb_path: int = 100000,
                                              headers: Optional[List[str]] = ('(A)', '(B)', '(C)', '(D)'),  # optimized for 2*2 figure
                                              is_log_strike_xaxis: bool = False,
                                              is_plot_vols: bool = True,
                                              figsize: Tuple[float, float] = plot.FIGSIZE,
                                              **kwargs
                                              ) -> plt.Figure:
        """
        comparision of model implied vols computed old_analytics under MMA and inverse measures vs mc pricer
        optimized for 2*2 figure
        """
        mma_label = 'MMA'
        inverse_lable = 'Inverse'

        model_prices_mma, model_ivols_mma = self.compute_chain_prices_with_vols(option_chain=option_chain, params=params,
                                                                                is_spot_measure=True,
                                                                                variable_type=variable_type,
                                                                                **kwargs)

        model_prices_inv, model_ivols_inv= self.compute_chain_prices_with_vols(option_chain=option_chain, params=params,
                                                                               is_spot_measure=False,
                                                                               variable_type=variable_type,
                                                                               **kwargs)

        # we perform MC simulation in MMA measure
        mc_kwargs = update_kwargs(kwargs, dict(is_spot_measure=True, variable_type=variable_type))
        model_prices_ttms, model_prices_ttms_ups, model_prices_ttms_downs, \
        mc_ivols, mc_ivols_up, mc_ivols_down, mc_stdev_ttms = self.compute_mc_chain_implied_vols(option_chain=option_chain,
                                                                                                 params=params,
                                                                                                 nb_path=nb_path,
                                                                                                 **mc_kwargs)

        if is_plot_vols:
            model_datas = {mma_label: model_ivols_mma, inverse_lable: model_ivols_inv}
            mc_data = mc_ivols
            mc_data_lower, mc_data_upper = mc_ivols_down, mc_ivols_up
        else:
            model_datas = {mma_label: model_prices_mma, inverse_lable: model_prices_inv}
            mc_data = model_prices_ttms
            mc_data_lower, mc_data_upper = model_prices_ttms_downs, model_prices_ttms_ups

        if option_chain.ttms.size < 4:
            nrows, ncols = 1, option_chain.ttms.size
        else:
            nrows, ncols = 2, option_chain.ttms.size//2

        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(nrows, ncols, figsize=figsize, tight_layout=True)

        for idx, ttm in enumerate(option_chain.ttms):
            if is_log_strike_xaxis:
                strikes = np.log(option_chain.strikes_ttms[idx] / option_chain.forwards[idx])
                xvar_format = '{:0.2f}'
                strike_name = 'log-strike'
            else:
                strikes = option_chain.strikes_ttms[idx]
                if variable_type == VariableType.LOG_RETURN:
                    xvar_format = '{:0,.2f}'
                    strike_name = 'strike'
                else:
                    xvar_format = '{:0.2f}'
                    strike_name = 'QVAR strike'

            model_vols = {}
            for key, model_data in model_datas.items():
                mse = np.sqrt(np.nanmean(np.power(model_data[idx] - mc_data[idx], 2)))
                model_vols[f"{key}: mse={mse:0.2%}"] = pd.Series(model_data[idx], index=strikes)
            model_vols = pd.DataFrame.from_dict(model_vols, orient='columns')

            if option_chain.ids is not None:
                if headers is not None:
                    title = f"{headers[idx]} slice - {option_chain.ids[idx]}"
                else:
                    title = f"slice - {option_chain.ids[idx]}"
            else:
                title = f"{ttm=:0.2f}"

            atm_vol = np.interp(x=option_chain.forwards[idx], xp=option_chain.strikes_ttms[idx],
                                fp=0.5 * (mc_data_lower[idx] + mc_data_upper[idx]))
            if is_log_strike_xaxis:
                atm_points = {'ATM': (0.0, atm_vol)}
            else:
                atm_points = {'ATM': (option_chain.forwards[idx], atm_vol)}

            plot.vol_slice_fit(bid_vol=pd.Series(mc_data_lower[idx], index=strikes),
                               ask_vol=pd.Series(mc_data_upper[idx], index=strikes),
                               model_vols=model_vols,
                               title=title,
                               bid_name='MC: -0.95ci',
                               ask_name='MC: +0.95ci',
                               strike_name=strike_name,
                               xvar_format=xvar_format,
                               x_rotation=0,
                               atm_points=atm_points,
                               ylabel='Implied vols' if is_plot_vols else 'Model prices',
                               yvar_format='{:.0%}' if is_plot_vols else '{:.2f}',
                               ax=axs[idx] if axs.ndim == 1 else axs[idx % 2][idx // 2],
                               **kwargs)
        return fig
