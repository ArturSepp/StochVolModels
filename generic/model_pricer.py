"""
ModelPricer is an interface class for the parent pricing model that defines the following interfaces:
1. price_chain is the abstract method that is implemented in the parent class for a specific model and its params using analytic solution
2. model_mc_price_chain is the interface method for pricing option chain using mc of model dynamics
3. calibrate_model_params_to_chain is the interface method that using model based price_chain
the rest of interface methods are concrete relying on price_chain
market options data is passed using data container ChainData
"""

# built in
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba.typed import List
from abc import ABC, abstractmethod
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, Optional

# internal
from generic.config import VariableType
import utils.plots as plot
from utils.funcs import set_seed

# generic
from generic.chain_data import ChainData

# set global mc seed, for resets call set_seed() locally
set_seed(7)


@dataclass
class ModelParams:
    pass


class ModelPricer(ABC):

    def __init__(self):
        super().__init__()

    #########################################################
    #                     generic interfaces
    #########################################################
    @abstractmethod
    def price_chain(self, chain_data: ChainData, params: ModelParams,
                    **kwargs
                    ) -> Tuple[np.ndarray, ...]:
        """
        abstract method for pricing chain data using model parameters
        recommended as a wrapper for numba implementation
        """
        pass

    def model_mc_price_chain(self, chain_data: ChainData, params: ModelParams, **kwargs) -> (
    List[np.ndarray], List[np.ndarray]):
        """
        abstract method for pricing chain data using simulation of model dynamics
        recommended as a wrapper for numba implementation
        """
        raise NotImplementedError(f"must be implemented in parent class")

    def calibrate_model_params_to_chain(self, chain_data: ChainData, **kwargs):
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
        chain_data = ChainData.slice_to_chain(ttm=ttm,
                                              forward=forward,
                                              strikes=strikes,
                                              optiontypes=optiontypes,
                                              discfactor=discfactor)
        model_prices = self.price_chain(chain_data=chain_data,
                                        params=params,
                                        **kwargs)
        model_ivols = chain_data.compute_model_ivols_from_chain_data(model_prices=model_prices)
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
    #          implemented interfaces for implied vol comptutions
    #########################################################
    def compute_model_ivols_for_chain(self,
                                      chain_data: ChainData,
                                      params: ModelParams,
                                      **kwargs
                                      ) -> np.ndarray:
        """
        price chain and compute model vols
        note that numba.List is equivalent type for Tuple
        """
        model_prices, model_ivols = self.compute_model_prices_and_ivols_for_chain(chain_data=chain_data,
                                                                                  params=params,
                                                                                  **kwargs)
        return model_ivols

    def compute_model_prices_and_ivols_for_chain(self,
                                                 chain_data: ChainData,
                                                 params: ModelParams,
                                                 **kwargs
                                                 ) -> (Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]):
        """
        price chain and compute model vols
        returns both prices and ivols
        note that numba.List is equivalent type for Tuple
        """
        model_prices = self.price_chain(chain_data=chain_data, params=params, **kwargs)
        model_ivols = chain_data.compute_model_ivols_from_chain_data(model_prices=model_prices)
        return model_prices, model_ivols

    def compute_mc_chain_implied_vols(self,
                                      chain_data: ChainData,
                                      params: ModelParams,
                                      nb_path: int = 100000,
                                      **kwargs
                                      ) -> (List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]):
        """
        use model_mc_price_chain intefrace to compute model ivols with confidence bound
        """
        model_prices_ttms, option_std_ttms = self.model_mc_price_chain(chain_data=chain_data,
                                                                       params=params,
                                                                       nb_path=nb_path,
                                                                       **kwargs)
        std_factor = 1.96
        model_prices_ttms_ups = List()
        model_prices_ttms_downs = List()
        for model_prices_ttm, option_std_ttm in zip(model_prices_ttms, option_std_ttms):
            model_prices_ttms_ups.append(model_prices_ttm + std_factor * option_std_ttm)
            model_prices_ttms_downs.append(np.maximum(model_prices_ttm - std_factor * option_std_ttm, 0.0))

        ivols_mid = chain_data.compute_model_ivols_from_chain_data(model_prices=model_prices_ttms)
        ivols_up = chain_data.compute_model_ivols_from_chain_data(model_prices=model_prices_ttms_ups)
        ivols_down = chain_data.compute_model_ivols_from_chain_data(model_prices=model_prices_ttms_downs)
        return model_prices_ttms, model_prices_ttms_ups, model_prices_ttms_downs, ivols_mid, ivols_up, ivols_down, option_std_ttms

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
                         chain_data: ChainData,
                         params: ModelParams,
                         is_log_strike_xaxis: bool = False,
                         **kwargs
                         ) -> plt.Figure:
        """
        plot model slice_t vols
        optimized for 2*2 figure
        """
        model_ivols = self.compute_model_ivols_for_chain(chain_data=chain_data, params=params, **kwargs)

        n_col = len(chain_data.ttms) // 2
        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(2, n_col, figsize=(18, 10), tight_layout=True)

        headers = ['(A)', '(B)', '(C)', '(D)']
        atm_vols = chain_data.get_chain_atm_vols()
        for idx, ttm in enumerate(chain_data.ttms):
            if is_log_strike_xaxis:
                strikes = np.log(chain_data.strikes_ttms[idx] / chain_data.forwards[idx])
                atm_forward = 0.0
                xvar_format = '{:0.2f}'
                strike_name = 'log-strike'
            else:
                strikes = chain_data.strikes_ttms[idx]
                atm_forward = chain_data.forwards[idx]
                xvar_format = '{:0,.0f}'
                strike_name = 'strike'

            midvols = 0.5 * (chain_data.bid_ivs[idx] + chain_data.ask_ivs[idx])
            mse2 = np.sqrt(np.nanmean(np.power(model_ivols[idx] - midvols, 2)))
            model_vols = pd.Series(model_ivols[idx], index=strikes, name=f"Model Fit: mse={mse2:0.2%}")
            if chain_data.ids is not None:
                title = f"{headers[idx]} slice - {chain_data.ids[idx]}"
            else:
                title = f"{ttm=:0.2f}"
            ax = axs[idx % 2][idx // 2]

            if is_log_strike_xaxis:
                atm_points = {'ATM': (0.0, atm_vols[idx])}
            else:
                atm_points = {'ATM': (atm_forward, atm_vols[idx])}

            plot.vol_slice_fit(bid_vol=pd.Series(chain_data.bid_ivs[idx], index=strikes),
                               ask_vol=pd.Series(chain_data.ask_ivs[idx], index=strikes),
                               model_vols=model_vols,
                               title=title,
                               atm_points=atm_points,
                               strike_name=strike_name,
                               xvar_format=xvar_format,
                               x_rotation=0,
                               ax=ax,
                               **kwargs)
        return fig

    def plot_model_ivols_vs_mc(self,
                               chain_data: ChainData,
                               params: ModelParams,
                               is_log_strike_xaxis: bool = False,
                               variable_type: VariableType = VariableType.LOG_RETURN,
                               nb_path: int = 100000,
                               **kwargs
                               ) -> None:
        """
        comparision of model implied vols computed old_analytics vs mc pricer
        optimized for 4*4 figure
        """
        model_ivols = self.compute_model_ivols_for_chain(chain_data=chain_data, params=params, **kwargs)

        mc_prices_ttms, _, _, mc_ivols, mc_ivols_up, mc_ivols_down, _ = self.compute_mc_chain_implied_vols(
            chain_data=chain_data,
            params=params,
            nb_path=nb_path,
            **kwargs)

        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(2, 2, figsize=(18, 10), tight_layout=True)

        for idx, ttm in enumerate(chain_data.ttms):
            if is_log_strike_xaxis:
                strikes = np.log(chain_data.strikes_ttms[idx] / chain_data.forwards[idx])
                xvar_format = '{:0.2f}'
                strike_name = 'log-strike'
            else:
                strikes = chain_data.strikes_ttms[idx]
                if variable_type == VariableType.LOG_RETURN:
                    xvar_format = '{:0,.0f}'
                    strike_name = 'strike'
                else:
                    xvar_format = '{:0.2f}'
                    strike_name = 'QVAR strike'

            mse2 = np.sqrt(np.nanmean(np.power(model_ivols[idx] - mc_ivols[idx], 2)))
            model_vol_t = pd.Series(model_ivols[idx], index=strikes, name=f"Model: mse={mse2:0.2%}")

            if chain_data.ids is not None:
                title = f"{chain_data.ids[idx]}, {ttm=:0.2f}"
            else:
                title = f"{ttm=:0.2f}"

            plot.vol_slice_fit(bid_vol=pd.Series(mc_ivols_down[idx], index=strikes),
                               ask_vol=pd.Series(mc_ivols_up[idx], index=strikes),
                               model_vols=model_vol_t,
                               title=title,
                               bid_name='MC: -0.95ci',
                               ask_name='MC: +0.95ci',
                               strike_name=strike_name,
                               xvar_format=xvar_format,
                               x_rotation=0,
                               ax=axs[idx % 2][idx // 2],
                               **kwargs)

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

    def plot_comp_mgf_with_mc(self,
                              chain_data: ChainData,
                              params: ModelParams,
                              is_log_strike_xaxis: bool = False,
                              variable_type: VariableType = VariableType.LOG_RETURN,
                              nb_path: int = 100000,
                              export_res: List[str] = None,
                              idx_ttm_to_export: Optional[int] = 2,
                              **kwargs
                              ) -> plt.Figure:
        """
        comparision of model implied vols computed old_analytics under MMA and inverse measures vs mc pricer
        optimized for 4*4 figure
        """
        kwargs = {'is_spot_measure': True,
                  'variable_type': VariableType.LOG_RETURN}
        afe2_prices_ttms = {}
        afe2_ivols = {}
        measures = ['MMA', 'INVERSE']
        selected_slices = None
        is_log_strike_xaxis = False

        export_res = ['put_call', 'ae2_price_inv', 'ae2_price_spot', 'mc_price_mid',
                      'diff_inv', 'diff_spot', 'stdevByZ']

        for measure in measures:
            if measure == 'INVERSE':
                kwargs['is_spot_measure'] = False
                chain_data.optiontypes_ttms = List([np.select([op_types == 'C', op_types == 'P'], ['IC', 'IP'], op_types) for
                                               op_types
                                               in chain_data.optiontypes_ttms])
            else:
                kwargs['is_spot_measure'] = True
                chain_data.optiontypes_ttms = List([np.select([op_types == 'IC', op_types == 'IP'], ['C', 'P'], op_types) for
                                               op_types
                                               in chain_data.optiontypes_ttms])

            model_prices, model_ivols = self.compute_model_prices_and_ivols_for_chain(chain_data=chain_data, params=params, **kwargs)
            afe2_prices_ttms[measure] = model_prices
            afe2_ivols[measure] = model_ivols

        # we perform MC simulation in MMA measure
        # TODO: check which measure (MMA vs inverse) is better numerically
        kwargs['is_spot_measure'] = True
        chain_data.optiontypes_ttms = List([np.select([op_types == 'IC', op_types == 'IP'], ['C', 'P'], op_types) for
                                            op_types in chain_data.optiontypes_ttms])

        model_prices_ttms, model_prices_ttms_ups, model_prices_ttms_downs, \
        mc_ivols, mc_ivols_up, mc_ivols_down, mc_stdev_ttms = self.compute_mc_chain_implied_vols(chain_data=chain_data,
                                                                                                 params=params,
                                                                                                 nb_path=nb_path,
                                                                                                 **kwargs)
        if export_res is not None and len(export_res) > 0:
            for idx, ttm in enumerate(chain_data.ttms):
                data = []
                z = 1.96
                for col in export_res:
                    if col == 'put_call': data.append(
                        pd.Series(chain_data.optiontypes_ttms[idx], index=chain_data.strikes_ttms[idx], name=f"Type"))
                    if col == 'ae2_price_spot': data.append(
                        pd.Series(afe2_prices_ttms['MMA'][idx], index=chain_data.strikes_ttms[idx], name=f"AE2:spot,fv"))
                    if col == 'ae2_ivol_spot': data.append(
                        pd.Series(afe2_ivols['MMA'][idx], index=chain_data.strikes_ttms[idx], name=f"AE2:spot,iv"))
                    if col == 'ae2_price_inv': data.append(
                        pd.Series(afe2_prices_ttms['INVERSE'][idx], index=chain_data.strikes_ttms[idx], name=f"AE2:inv,fv"))
                    if col == 'ae2_ivol_inv': data.append(
                        pd.Series(afe2_ivols['INVERSE'][idx], index=chain_data.strikes_ttms[idx], name=f"AE2:inv,fv"))
                    if col == 'mc_price_mid': data.append(
                        pd.Series(model_prices_ttms[idx], index=chain_data.strikes_ttms[idx], name=f"MC:mid,fv"))
                    if col == 'mc_ivol_mid': data.append(
                        pd.Series(mc_ivols[idx], index=chain_data.strikes_ttms[idx], name=f"MC:mid,iv"))
                    if col == 'mc_ivol_up': data.append(
                        pd.Series(mc_ivols_up[idx], index=chain_data.strikes_ttms[idx], name=f"MC:up,iv"))
                    if col == 'mc_ivol_down': data.append(
                        pd.Series(mc_ivols_down[idx], index=chain_data.strikes_ttms[idx], name=f"MC:dw,iv"))
                    if col == 'diff_spot': data.append(
                        pd.Series(afe2_prices_ttms['MMA'][idx] - model_prices_ttms[idx], index=chain_data.strikes_ttms[idx],
                                  name=f"Diff,spot").map('{:,.2e}'.format))
                    if col == 'diff_inv': data.append(
                        pd.Series(afe2_prices_ttms['INVERSE'][idx] - model_prices_ttms[idx], index=chain_data.strikes_ttms[idx],
                                  name=f"Diff,inv").map('{:,.2e}'.format))
                    if col == 'stdev': data.append(
                        pd.Series(mc_stdev_ttms[idx], index=chain_data.strikes_ttms[idx], name=f"StdDev").map('{:,.2e}'.format))
                    if col == 'stdevByZ': data.append(
                        pd.Series(mc_stdev_ttms[idx] * z, index=chain_data.strikes_ttms[idx], name=f"StdDev*z").map(
                            '{:,.2e}'.format))
                data = pd.concat(data, axis=1)
                # suffix = str(uuid.uuid4().hex)
                # data.to_latex(f"c:/temp/ttm-{idx}-{suffix}.tex")
                if idx_ttm_to_export is not None and idx == idx_ttm_to_export:
                    data.to_latex(f"c:/temp/mc_vs_analytic_{chain_data.ticker}_ttm-{chain_data.ids[idx]}.tex")

        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(2, 2, figsize=(18, 10), tight_layout=True)

        headers = ['(A)', '(B)', '(C)', '(D)']
        for idx, ttm in enumerate(chain_data.ttms):
            if is_log_strike_xaxis:
                strikes = np.log(chain_data.strikes_ttms[idx] / chain_data.forwards[idx])
                xvar_format = '{:0.2f}'
                strike_name = 'log-strike'
            else:
                strikes = chain_data.strikes_ttms[idx]
                if variable_type == VariableType.LOG_RETURN:
                    xvar_format = '{:0,.2f}'
                    strike_name = 'strike'
                else:
                    xvar_format = '{:0.2f}'
                    strike_name = 'QVAR strike'

            mse2 = {}
            model_vols = {}
            for measure in measures:
                measureUpper = measure.upper()
                mse2[measureUpper] = np.sqrt(np.nanmean(np.power(afe2_ivols[measureUpper][idx] - mc_ivols[idx], 2)))
                model_vols[measureUpper] = pd.Series(afe2_ivols[measureUpper][idx], index=strikes,
                                                     name=f"Second-order, MMA: mse={mse2[measureUpper]:0.2%}")
            model_vols = pd.concat(model_vols, axis=1)

            if chain_data.ids is not None:
                title = f"{headers[idx]} slice - {chain_data.ids[idx]}"
            else:
                title = f"{ttm=:0.2f}"

            atm_vol = np.interp(x=chain_data.forwards[idx], xp=chain_data.strikes_ttms[idx],
                                fp=0.5 * (mc_ivols_down[idx] + mc_ivols_up[idx]))
            if is_log_strike_xaxis:
                atm_points = {'ATM': (0.0, atm_vol)}
            else:
                atm_points = {'ATM': (chain_data.forwards[idx], atm_vol)}

            plot.vol_slice_fit(bid_vol=pd.Series(mc_ivols_down[idx], index=strikes),
                               ask_vol=pd.Series(mc_ivols_up[idx], index=strikes),
                               model_vols=model_vols,
                               title=title,
                               bid_name='MC: -0.95ci',
                               ask_name='MC: +0.95ci',
                               strike_name=strike_name,
                               xvar_format=xvar_format,
                               x_rotation=0,
                               atm_points=atm_points,
                               ax=axs[idx % 2][idx // 2],
                               **kwargs)
        return fig
