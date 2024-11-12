"""
data container for option chain
data is provided as:
1 arrays of ttms, forwards, discounts
2 lists of arrays with strikes, optiom types and bid / ask prices and vols
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

import pandas as pd
from numba.typed import List

import stochvolmodels.pricers.analytic.bsm as bsm
from  stochvolmodels.utils.var_swap_pricer import compute_var_swap_strike
from stochvolmodels.pricers.factor_hjm.rate_core import get_default_swap_term_structure, swap_rate
import stochvolmodels.pricers.analytic.bachelier as bachel


@dataclass
class OptionSlice:
    """
    container for slice data
    """
    ttm: float
    forward: float
    strikes: np.ndarray
    optiontypes: np.ndarray
    id: str
    discfactor: float = None  # discount factors
    discount_rate: float = None  # discount rates
    bid_ivs: Optional[np.ndarray] = None
    ask_ivs: Optional[np.ndarray] = None
    bid_prices: Optional[np.ndarray] = None
    ask_prices: Optional[np.ndarray] = None

    def __post_init__(self):
        """
        to do: check dimension aligmnent
        make consistent discfactors
        """
        if self.discfactor is not None:
            self.discount_rate = - np.log(self.discfactor) / self.ttm
        elif self.discount_rate is not None:
            self.discfactor = np.exp(-self.discount_rate * self.ttm)
        else:  # use zeros
            self.discfactor = 1.0
            self.discount_rate = 0.0


@dataclass
class OptionChain:
    """
    container for chain data
    note we do not use chain as list of slices here
    for extensive use of numba we use List[np.ndarray] with per slice data
    """
    ttms: np.ndarray
    forwards: np.ndarray
    strikes_ttms: List[np.ndarray]
    optiontypes_ttms: List[np.ndarray]
    ids: Optional[np.ndarray]  # slice_t names
    discfactors: Optional[np.ndarray] = None  # discount factors
    discount_rates: Optional[np.ndarray] = None  # discount rates
    ticker: Optional[str] = None  # associated ticker
    bid_ivs: Optional[List[np.ndarray]] = None
    ask_ivs: Optional[List[np.ndarray]] = None
    bid_prices: Optional[List[np.ndarray]] = None
    ask_prices: Optional[List[np.ndarray]] = None
    forwards0: Optional[np.ndarray] = None  # when we need to normalize options price

    def __post_init__(self):
        """
        to do: check dimension aligmnent
        make consistent discfactors
        """
        if self.discfactors is not None:
            self.discount_rates = - np.log(self.discfactors) / self.ttms
        elif self.discount_rates is not None:
            self.discfactors = np.exp(-self.discount_rates * self.ttms)
        else:  # use zeros
            self.discfactors = np.ones_like(self.ttms)
            self.discount_rates = np.zeros_like(self.ttms)

    def print(self) -> None:
        this = dict(ttms=self.ttms,
                    forwards=self.forwards,
                    strikes_ttms=self.strikes_ttms,
                    optiontypes_ttms=self.optiontypes_ttms,
                    ids=self.ids,
                    bid_ivs=self.bid_ivs,
                    ask_ivs=self.ask_ivs)
        for k, v in this.items():
            print(f"{k}:\n{v}")

    @classmethod
    def slice_to_chain(cls,
                       ttm: float,
                       forward: float,
                       strikes: np.ndarray,
                       optiontypes: np.ndarray,
                       discfactor: float = 1.0,
                       id: Optional[str] = None
                       ) -> OptionChain:

        return cls(ttms=np.array([ttm]),
                   forwards=np.array([forward]),
                   strikes_ttms=(strikes,),
                   optiontypes_ttms=(optiontypes,),
                   discfactors=np.array([discfactor]),
                   ids=np.array([id]) if id is not None else np.array([f"{ttm:0.2f}"]))

    def get_mid_vols(self) -> List[np.ndarray]:
        if self.bid_ivs is not None and self.ask_ivs is not None:
            return List(0.5 * (bid_iv + ask_iv) for bid_iv, ask_iv in zip(self.bid_ivs, self.ask_ivs))
        else:
            return None

    def get_chain_deltas(self) -> List[np.ndarray]:
        deltas_ttms = bsm.compute_bsm_vanilla_deltas_ttms(ttms=self.ttms,
                                                          forwards=self.forwards,
                                                          strikes_ttms=self.strikes_ttms,
                                                          optiontypes_ttms=self.optiontypes_ttms,
                                                          vols_ttms=self.get_mid_vols())
        return deltas_ttms

    def get_chain_vegas(self, is_unit_ttm_vega: bool = False) -> List[np.ndarray]:
        if is_unit_ttm_vega:
            ttms = np.ones_like(self.ttms)
        else:
            ttms = self.ttms
        vegas_ttms = bsm.compute_bsm_vegas_ttms(ttms=ttms,
                                                forwards=self.forwards,
                                                strikes_ttms=self.strikes_ttms,
                                                optiontypes_ttms=self.optiontypes_ttms,
                                                vols_ttms=self.get_mid_vols())
        return vegas_ttms

    def get_chain_atm_vols(self) -> np.ndarray:
        atm_vols = np.zeros(len(self.ttms))
        for idx, (forward, strikes_ttm, y) in enumerate(zip(self.forwards, self.strikes_ttms, self.get_mid_vols())):
            atm_vols[idx] = np.interp(x=forward, xp=strikes_ttm, fp=y)
        return atm_vols

    def get_chain_skews(self, delta: float = 0.25) -> np.ndarray:
        skews = np.zeros(len(self.ttms))
        deltas_ttms = self.get_chain_deltas()
        for idx, (deltas, vols) in enumerate(zip(deltas_ttms, self.get_mid_vols())):
            dput = np.interp(x=-delta, xp=deltas, fp=vols)
            d50 = np.interp(x=0.5, xp=deltas, fp=vols)
            dcall= np.interp(x=delta, xp=deltas, fp=vols)
            skews[idx] = (dput - dcall)/d50
        return skews

    def get_chain_data_as_xy(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        these data are needed for to pass x and y for model calibrations
        """
        mid_vols = List(0.5 * (bid_iv + ask_iv) for bid_iv, ask_iv in zip(self.bid_ivs, self.ask_ivs))
        x = (self.ttms, self.forwards, self.discfactors, self.strikes_ttms, self.optiontypes_ttms)
        y = mid_vols
        return x, y

    def compute_model_ivols_from_chain_data(self,
                                            model_prices: List[np.ndarray],
                                            forwards: np.ndarray = None
                                            ) -> List[np.ndarray]:
        if forwards is None:
            forwards = self.forwards

        model_ivols = bsm.infer_bsm_ivols_from_model_chain_prices(ttms=self.ttms,
                                                                  forwards=forwards,
                                                                  discfactors=self.discfactors,
                                                                  strikes_ttms=self.strikes_ttms,
                                                                  optiontypes_ttms=self.optiontypes_ttms,
                                                                  model_prices_ttms=model_prices)
        return model_ivols

    @classmethod
    def to_forward_normalised_strikes(cls, obj: OptionChain) -> OptionChain:
        """
        strike and prices to normalized by forwards
        """
        new_strikes_ttms = List()
        for strikes_ttm, forward in zip(obj.strikes_ttms, obj.forwards):
            new_strikes = strikes_ttm / forward
            new_strikes_ttms.append(new_strikes)

        return cls(ttms=obj.ttms,
                   forwards=np.ones_like(obj.forwards),
                   strikes_ttms=new_strikes_ttms,
                   optiontypes_ttms=obj.optiontypes_ttms,
                   discfactors=obj.discfactors,
                   ticker=obj.ticker,
                   ids=obj.ids,
                   bid_ivs=obj.bid_ivs, ask_ivs=obj.ask_ivs,
                   forwards0=obj.forwards)

    @classmethod
    def to_uniform_strikes(cls, obj: OptionChain, num_strikes: int = 21) -> OptionChain:
        """
        in some situations (like model price display) we want to get a uniform grid corresponding to the chain
        bid_ivs and ask_ivs will be set to none
        """
        new_strikes_ttms = List()
        new_optiontypes_ttms = List()
        for strikes_ttm, forward in zip(obj.strikes_ttms, obj.forwards):
            new_strikes = np.linspace(strikes_ttm[0], strikes_ttm[-1], num_strikes)
            new_strikes_ttms.append(new_strikes)
            new_optiontypes_ttms.append(np.where(new_strikes >= forward, 'C', 'P'))

        return cls(ttms=obj.ttms, forwards=obj.forwards, strikes_ttms=new_strikes_ttms,
                   optiontypes_ttms=new_optiontypes_ttms, discfactors=obj.discfactors,
                   ticker=obj.ticker,
                   ids=obj.ids,
                   bid_ivs=None, ask_ivs=None)

    def get_slice(self, id: str) -> OptionSlice:
        idx = list(self.ids).index(id)
        option_slice = OptionSlice(id=self.ids[idx],
                                   ttm=self.ttms[idx],
                                   forward=self.forwards[idx],
                                   strikes=self.strikes_ttms[idx],
                                   optiontypes=self.optiontypes_ttms[idx],
                                   discfactor=self.discfactors[idx],
                                   bid_ivs=None if self.bid_ivs is None else self.bid_ivs[idx],
                                   ask_ivs=None if self.ask_ivs is None else self.ask_ivs[idx],
                                   bid_prices=None if self.bid_prices is None else self.bid_prices[idx],
                                   ask_prices=None if self.ask_prices is None else self.ask_prices[idx])
        return option_slice

    def get_slice_varswap_strikes(self, floor_with_atm_vols: bool = True) -> pd.Series:
        varswap_strikes = np.zeros_like(self.ttms)
        vols_ttms = self.get_mid_vols()
        for idx, ttm in enumerate(self.ttms):
            mid_prices = bsm.compute_bsm_vanilla_slice_prices(ttm=ttm,
                                                              forward=self.forwards[idx],
                                                              strikes=self.strikes_ttms[idx],
                                                              vols=vols_ttms[idx],
                                                              optiontypes=self.optiontypes_ttms[idx])
            strikes = self.strikes_ttms[idx]
            puts_cond = self.optiontypes_ttms[idx] == 'P'
            puts = pd.Series(mid_prices[puts_cond], index=strikes[puts_cond])
            calls = pd.Series(mid_prices[puts_cond == False], index=strikes[puts_cond == False])
            varswap_strikes[idx] = compute_var_swap_strike(puts=puts, calls=calls, forward=self.forwards[idx], ttm=ttm)

        if floor_with_atm_vols:
            varswap_strikes = np.maximum(self.get_chain_atm_vols(), varswap_strikes)

        return pd.Series(varswap_strikes, index=self.ttms)

    @classmethod
    def get_slices_as_chain(cls, option_chain: OptionChain, ids: List[str]) -> OptionChain:
        """
        return a subset of chain for given ids
        """
        if len(ids) == 1:
            idx = list(option_chain.ids).index(ids[0])  # get location of ids
            option_chain = cls(ids=ids,
                               ttms=np.array([option_chain.ttms[idx]]),
                               ticker=option_chain.ticker,
                               forwards=np.array([option_chain.forwards[idx]]),
                               strikes_ttms=List([option_chain.strikes_ttms[idx]]),
                               optiontypes_ttms=List([option_chain.optiontypes_ttms[idx]]),
                               discfactors=np.array([option_chain.discfactors[idx]]),
                               bid_ivs=None if option_chain.bid_ivs is None else List([option_chain.bid_ivs[idx]]),
                               ask_ivs=None if option_chain.ask_ivs is None else List([option_chain.ask_ivs[idx]]),
                               bid_prices=None if option_chain.bid_prices is None else List([option_chain.bid_prices[idx]]),
                               ask_prices=None if option_chain.ask_prices is None else List([option_chain.ask_prices[idx]]))
        else:
            indices = np.isin(option_chain.ids, ids).nonzero()[0]
            option_chain = cls(ids=ids,
                               ttms=option_chain.ttms[indices],
                               ticker=option_chain.ticker,
                               forwards=option_chain.forwards[indices],
                               strikes_ttms=List(option_chain.strikes_ttms[idx] for idx in indices),
                               optiontypes_ttms=List(option_chain.optiontypes_ttms[idx] for idx in indices),
                               discfactors=option_chain.discfactors[indices],
                               bid_ivs=None if option_chain.bid_ivs is None else List(option_chain.bid_ivs[idx] for idx in indices),
                               ask_ivs=None if option_chain.ask_ivs is None else List(option_chain.ask_ivs[idx] for idx in indices),
                               bid_prices=None if option_chain.bid_prices is None else List(option_chain.bid_prices[idx] for idx in indices),
                               ask_prices=None if option_chain.ask_prices is None else List(option_chain.ask_prices[idx] for idx in indices))
        return option_chain

    @classmethod
    def get_uniform_chain(cls,
                          ttms: np.ndarray = np.array([0.083, 0.25]),
                          ids: np.ndarray = np.array(['1m', '3m']),
                          forwards: np.ndarray = np.array([1.0, 1.0]),
                          strikes: np.ndarray = np.linspace(0.9, 1.1, 3),
                          flat_vol: float = 0.2
                          ) -> OptionChain:
        return cls(ttms=ttms,
                   ids=ids,
                   forwards=forwards,
                   strikes_ttms=List([strikes for _ in ttms]),
                   bid_ivs=List([flat_vol*np.ones_like(strikes) for _ in ttms]),
                   ask_ivs=List([flat_vol*np.ones_like(strikes) for _ in ttms]),
                   optiontypes_ttms=List([np.where(strikes >= forward, 'C', 'P') for forward in forwards]))


@dataclass
class SwOptionChain:
    ccy: str
    ttms: np.ndarray  # swaption expiries
    tenors: np.ndarray
    ttms_ids: List[str]  # id of expiries, e.g. ['1y', '2y', '3y', '5y', '7y', '10y']
    tenors_ids: List[str]  # id of tenors, e.g. ['10y']
    forwards: List[np.ndarray]
    strikes_ttms: List[List[np.ndarray]]
    bid_ivs: List[List[np.ndarray]]
    ask_ivs: List[List[np.ndarray]]
    ticker: Optional[str] = None  # associated ticker

    def __post_init__(self):
        assert self.ttms.size == len(self.ttms_ids)
        assert self.tenors.size == len(self.tenors_ids)
        assert np.all(np.diff(self.ttms) >= 0) and np.all(self.ttms >= 0)  # check that expiries are sorted and positive
        assert np.all(np.diff(self.tenors) >= 0) and np.all(
            self.tenors >= 0)  # check that tenors are sorted and positive
        self.optiontypes_ttms = tuple([np.repeat('C', self.strikes_ttms[0][0].size) for ttm in
                                       self.ttms])  # np.where(pstrikes < 0.0, 'P', 'C').astype(str)
        # first dimension for tenors
        assert len(self.strikes_ttms) == len(self.tenors_ids)
        assert len(self.bid_ivs) == len(self.ask_ivs) == len(self.tenors_ids)
        # second dimesion for expiries
        assert len(self.strikes_ttms[0]) == len(self.ttms_ids)
        assert len(self.bid_ivs[0]) == len(self.ask_ivs[0]) == len(self.ttms_ids)
        # for fixed tenor and expiry, strikes must be 1D
        assert self.strikes_ttms[0][0].ndim == 1
        # forward must be of size # tenors x # expiries
        assert len(self.forwards) == len(self.tenors_ids) and self.forwards[0].size == len(self.ttms_ids)
        # strikes have same dimension and are consistent with vols
        for i, _ in enumerate(self.tenors_ids):
            for j, _ in enumerate(self.ttms_ids):
                assert self.strikes_ttms[i][j].size == self.strikes_ttms[0][0].size
                assert self.bid_ivs[i][j].size == self.ask_ivs[0][0].size
                assert self.strikes_ttms[i][j].size == self.bid_ivs[0][0].size

    @classmethod
    def create_swaption_chain_MF(cls,
                                 ccy: str,
                                 tenors: np.ndarray,
                                 tenors_ids: List[str],
                                 ttms: np.ndarray,
                                 ttms_ids: List[str],
                                 forwards: List[np.ndarray],
                                 strikes_ttms: List[List[np.ndarray]],
                                 ivs: List[List[np.ndarray]],
                                 ticker: str) -> SwOptionChain:

        # re-center strikes as we work with flat zero curve
        for idx_tenor, tenor in enumerate(tenors):
            for idx_ttm, ttm in enumerate(ttms):
                ts_sw = get_default_swap_term_structure(ttm, tenor)
                strikes_ttms[idx_tenor][idx_ttm] = strikes_ttms[idx_tenor][idx_ttm] - forwards[idx_tenor][idx_ttm] + swap_rate(ccy, ttm, ts_sw)
                forwards[idx_tenor][idx_ttm] = swap_rate(ccy, ttm, ts_sw)
                # print(f"idx_tenor={idx_tenor}, idx_ttm={idx_ttm}: OK")

        data = SwOptionChain(ccy=ccy,
                             ttms=ttms,
                             tenors=tenors,
                             ttms_ids=ttms_ids,
                             tenors_ids=tenors_ids,
                             forwards=forwards,
                             strikes_ttms=strikes_ttms,
                             bid_ivs=ivs,
                             ask_ivs=ivs,
                             ticker=ticker)

        return data

    def reduce_strikes(self, nb_otms: int):
        nb_strikes = int((self.strikes_ttms[0][0].size - 1) / 2)
        if nb_otms > nb_strikes:
            raise ValueError(f"number of strikes ={nb_otms} to reduce is > number of otm strikes ={nb_strikes}")
        range_ = range(nb_strikes - nb_otms, nb_strikes + nb_otms + 1)
        strikes = [[self.strikes_ttms[idx_tenor][idx_ttm][range_] for idx_ttm, _ in enumerate(self.ttms_ids)] for
                   idx_tenor, _ in enumerate(self.tenors_ids)]
        bid_ivs = [[self.bid_ivs[idx_tenor][idx_ttm][range_] for idx_ttm, _ in enumerate(self.ttms_ids)] for
                   idx_tenor, _ in enumerate(self.tenors_ids)]
        ask_ivs = [[self.ask_ivs[idx_tenor][idx_ttm][range_] for idx_ttm, _ in enumerate(self.ttms_ids)] for
                   idx_tenor, _ in enumerate(self.tenors_ids)]

        chain = SwOptionChain(ccy=self.ccy,
                              ttms=self.ttms,
                              tenors=self.tenors,
                              ttms_ids=self.ttms_ids,
                              tenors_ids=self.tenors_ids,
                              forwards=self.forwards,
                              strikes_ttms=strikes,
                              bid_ivs=bid_ivs,
                              ask_ivs=ask_ivs,
                              ticker=self.ticker)
        return chain

    def reduce_ttms(self, ttms_ids: List[str]):
        if not np.all(np.isin(ttms_ids, self.ttms_ids)):
            raise ValueError(f"Expiries to be removed not present if chain")
        idx_ttms = np.where(np.isin(self.ttms_ids, ttms_ids))[0]
        strikes = [[self.strikes_ttms[idx_tenor][idx_ttm] for idx_ttm in idx_ttms] for
                   idx_tenor, _ in enumerate(self.tenors_ids)]
        bid_ivs = [[self.bid_ivs[idx_tenor][idx_ttm] for idx_ttm in idx_ttms] for
                   idx_tenor, _ in enumerate(self.tenors_ids)]
        ask_ivs = [[self.ask_ivs[idx_tenor][idx_ttm] for idx_ttm in idx_ttms] for
                   idx_tenor, _ in enumerate(self.tenors_ids)]
        forwards = [np.array([self.forwards[idx_tenor][idx_ttm] for idx_ttm in idx_ttms]) for
                   idx_tenor, _ in enumerate(self.tenors_ids)]

        chain = SwOptionChain(ccy=self.ccy,
                              ttms=self.ttms[idx_ttms],
                              tenors=self.tenors,
                              ttms_ids=ttms_ids,
                              tenors_ids=self.tenors_ids,
                              forwards=forwards,
                              strikes_ttms=strikes,
                              bid_ivs=bid_ivs,
                              ask_ivs=ask_ivs,
                              ticker=self.ticker)
        return chain

    def reduce_tenors(self, tenors_ids: List[str]):
        if not np.all(np.isin(tenors_ids, self.tenors_ids)):
            raise ValueError(f"Tenors to be removed not present if chain")
        idx_tenors = np.where(np.isin(self.tenors_ids, tenors_ids))[0]
        strikes = [[self.strikes_ttms[idx_tenor][idx_ttm] for idx_ttm, _ in enumerate(self.ttms_ids)] for
                   idx_tenor in idx_tenors]
        bid_ivs = [[self.bid_ivs[idx_tenor][idx_ttm] for idx_ttm, _ in enumerate(self.ttms_ids)] for
                   idx_tenor in idx_tenors]
        ask_ivs = [[self.ask_ivs[idx_tenor][idx_ttm] for idx_ttm, _ in enumerate(self.ttms_ids)] for
                   idx_tenor in idx_tenors]
        forwards = [np.array([self.forwards[idx_tenor][idx_ttm] for idx_ttm, _ in enumerate(self.ttms_ids)]) for
                   idx_tenor in idx_tenors]

        chain = SwOptionChain(ccy=self.ccy,
                              ttms=self.ttms,
                              tenors=self.tenors[idx_tenors],
                              ttms_ids=self.ttms_ids,
                              tenors_ids=[self.tenors_ids[idx] for idx in idx_tenors],
                              forwards=forwards,
                              strikes_ttms=strikes,
                              bid_ivs=bid_ivs,
                              ask_ivs=ask_ivs,
                              ticker=self.ticker)
        return chain

    def get_chain_atm_vols(self) -> List[np.ndarray]:
        atm_vols = List()
        for idx_tenor, (forwards_tenor, strikes_tenor, vols_tenor) in enumerate(zip(self.forwards, self.strikes_ttms, self.get_mid_vols())):
            atm_vols_tenor = np.zeros_like(forwards_tenor)
            for idx, (forward, strikes, vol_slice) in enumerate(zip(forwards_tenor, strikes_tenor, vols_tenor)):
                atm_vols_tenor[idx] = np.interp(x=forward, xp=strikes, fp=vol_slice)
            atm_vols.append(np.array(atm_vols_tenor))
        return atm_vols

    def get_mid_vols(self) -> List[List[np.ndarray]]:
        return [[0.5 * (self.bid_ivs[idx_tenor][idx_ttm] + self.ask_ivs[idx_tenor][idx_ttm]) for
                 idx_ttm, _ in enumerate(self.ttms_ids)] for idx_tenor, _ in enumerate(self.tenors_ids)]

    # def get_chain_deltas(self) -> List[np.ndarray]:
    #     deltas_ttms = bsm.compute_bsm_deltas_ttms(ttms=self.ttms,
    #                                               forwards=self.forwards,
    #                                               strikes_ttms=self.strikes_ttms,
    #                                               optiontypes_ttms=self.optiontypes_ttms,
    #                                               vols_ttms=self.get_mid_vols())
    #     return deltas_ttms

    def get_chain_vegas(self, is_unit_ttm_vega: bool = False) -> List[List[np.ndarray]]:
        if is_unit_ttm_vega:
            ttms = np.ones_like(self.ttms)
        else:
            ttms = self.ttms
        vegas_chain = List()
        for idx_tenor, (forwards, strikes_ttms, mid_vols) in enumerate(zip(self.forwards, self.strikes_ttms, self.get_mid_vols())):
            vegas_ttms = bachel.compute_normal_vegas_ttms(ttms=ttms,
                                                          forwards=forwards,
                                                          strikes_ttms=tuple(strikes_ttms),
                                                          optiontypes_ttms=self.optiontypes_ttms,
                                                          vols_ttms=tuple(mid_vols))
            vegas_chain.append(vegas_ttms)
        return vegas_chain

    def compute_model_ivols_from_chain_data(self, model_prices: List[np.ndarray]) -> List[np.ndarray]:
        model_ivols = bsm.infer_bsm_ivols_from_model_chain_prices(ttms=self.ttms,
                                                                  forwards=self.forwards,
                                                                  discfactors=self.discfactors,
                                                                  strikes_ttms=self.strikes_ttms,
                                                                  optiontypes_ttms=self.optiontypes_ttms,
                                                                  model_prices_ttms=model_prices)
        return model_ivols

    @classmethod
    def get_slices_as_chain(cls, option_chain: SwOptionChain, ids: List[str]) -> SwOptionChain:
        indices = np.in1d(option_chain.ttms_ids, ids).nonzero()[0]
        option_chain = cls(ccy=option_chain.ccy,
                           ttms=option_chain.ttms[indices],
                           tenors=option_chain.tenors,
                           ttms_ids=option_chain.ttms_ids[indices],
                           tenors_ids=option_chain.tenors_ids,
                           forwards=[option_chain.forwards[j][indices] for j, _ in enumerate(option_chain.tenors_ids)],
                           strikes_ttms=[[option_chain.strikes_ttms[j][idx] for idx in indices] for j, _ in
                                         enumerate(option_chain.tenors_ids)],
                           ticker=option_chain.ticker,
                           bid_ivs=[[option_chain.bid_ivs[j][idx] for idx in indices] for j, _ in
                                    enumerate(option_chain.tenors_ids)],
                           ask_ivs=[[option_chain.ask_ivs[j][idx] for idx in indices] for j, _ in
                                    enumerate(option_chain.tenors_ids)])
        return option_chain

    @classmethod
    def remap_to_inc_delta(cls, vols: pd.Series) -> pd.Series:
        vols.index = [-x for x in vols.index]
        return vols


    @classmethod
    def remap_to_pc_delta(cls, inc_grid: np.ndarray) -> np.ndarray:
        put_cond = inc_grid < -0.5
        call_cond = inc_grid >= -0.5
        put_grid = -inc_grid[put_cond] - 1.0
        call_grid = -inc_grid[call_cond]
        grid = np.concatenate((put_grid, call_grid))

        return grid


@dataclass
class FutOptionChain:
    """
    container for chain data
    note we do not use chain as list of slices here
    for extensive use of numba we use List[np.ndarray] with per slice data
    """
    ccy: str
    ttms: np.ndarray
    forwards: np.ndarray
    strikes_ttms: List[np.ndarray]
    ttms_ids: Optional[np.ndarray]  # slice_t names
    ivs_call_ttms: List[np.ndarray]  # implied vol of call options on futures
    ivs_put_ttms: List[np.ndarray]  # implied vol of put options on futures
    ticker: Optional[str] = None  # associated ticker
    call_oi: List[np.ndarray] = None  # open interest of call options
    put_oi: List[np.ndarray] = None  # open interest of put options
    call_vol: List[np.ndarray] = None  # volume of call options
    put_vol: List[np.ndarray] = None  # volume of call options


    def __post_init__(self):
        assert self.ttms.size == len(self.ttms_ids)
        assert np.all(np.diff(self.ttms) >= 0) and np.all(self.ttms >= 0)  # check that expiries are sorted and positive
        self.optiontypes_ttms = tuple([np.repeat('C', self.strikes_ttms[idx_ttm].size) for idx_ttm, ttm in enumerate(self.ttms)])

        assert all(ivs_call_ttm.shape == ivs_put_ttm.shape for ivs_call_ttm, ivs_put_ttm in zip(self.ivs_call_ttms, self.ivs_put_ttms))

        assert len(self.ivs_call_ttms) == self.ttms.size
        assert self.ttms.shape == self.forwards.shape
        # strikes must be 1D
        assert all(strikes.ndim == 1 for strikes in self.strikes_ttms)
        for ivs_call_ttm, strikes_ttm in zip(self.ivs_call_ttms, self.strikes_ttms):
            assert ivs_call_ttm.shape == strikes_ttm.shape
        # put open interest call open interest must be specified (opr not specified) simulataneously
        assert (self.call_oi is not None and self.put_oi is not None) or (self.call_oi is None and self.put_oi is None)
        if self.call_oi is not None:
            for ivs_call_ttm, call_oi_ttm, put_oi_ttm in zip(self.ivs_call_ttms, self.call_oi, self.put_oi):
                assert ivs_call_ttm.shape == call_oi_ttm.shape == put_oi_ttm.shape

        assert (self.call_vol is not None and self.put_vol is not None) or (self.call_vol is None and self.put_vol is None)
        if self.call_vol is not None:
            for ivs_call_ttm, call_vol_ttm, put_vol_ttm in zip(self.ivs_call_ttms, self.call_vol, self.put_vol):
                assert ivs_call_ttm.shape == call_vol_ttm.shape == put_vol_ttm.shape

    def filter_by_oi(self,
                     max_strikes: int,
                     include_atm: bool) -> FutOptionChain:
        if self.call_oi is None:
            raise NotImplementedError(f"call/put open interest cannot be None")

        mid_idx = int(0.5 * (self.strikes_ttms[0].size - 1))

        strikes_oi_ttms = []
        ivs_call_oi_ttms = []
        ivs_put_oi_ttms = []
        call_oi_ttms = []
        put_oi_ttms = []

        for idx_ttm, ttm in enumerate(self.ttms):
            oi_ttm = self.call_oi[idx_ttm] + self.put_oi[idx_ttm]
            # top N options by open interest
            idxs = oi_ttm.argsort()[-max_strikes:][::-1]
            if include_atm and mid_idx not in idxs:
                raise ValueError(f"atm strike not found among top {max_strikes} liquid options")
            idxs_sort = np.sort(idxs)
            strikes_oi_ttm = self.strikes_ttms[idx_ttm][idxs_sort]
            ivs_call_oi_ttm = self.ivs_call_ttms[idx_ttm][idxs_sort]
            ivs_put_oi_ttm = self.ivs_put_ttms[idx_ttm][idxs_sort]
            call_oi_ttm = self.call_oi[idx_ttm][idxs_sort]
            put_oi_ttm = self.put_oi[idx_ttm][idxs_sort]

            strikes_oi_ttms.append(strikes_oi_ttm)
            ivs_call_oi_ttms.append(ivs_call_oi_ttm)
            ivs_put_oi_ttms.append(ivs_put_oi_ttm)
            call_oi_ttms.append(call_oi_ttm)
            put_oi_ttms.append(put_oi_ttm)

        fut_chain_oi = FutOptionChain(ccy=self.ccy,
                                      ttms=self.ttms,
                                      forwards=self.forwards,
                                      strikes_ttms=np.array(strikes_oi_ttms),
                                      ivs_call_ttms=np.array(ivs_call_oi_ttms),
                                      ivs_put_ttms=np.array(ivs_put_oi_ttms),
                                      ttms_ids=self.ttms_ids,
                                      call_oi=call_oi_ttms,
                                      put_oi=put_oi_ttms,
                                      ticker=self.ticker)

        return fut_chain_oi




    def get_mid_vols(self) -> List[np.ndarray]:
        return self.ivs_call_ttms

    def get_chain_vegas(self):
        optiontypes_ttms = np.repeat('C', self.strikes_ttms[0].size)

        vegas_ttms = bachel.compute_normal_vegas_ttms(ttms=self.ttms,
                                                      forwards=self.forwards,
                                                      strikes_ttms=self.strikes_ttms,
                                                      optiontypes_ttms=tuple(optiontypes_ttms),
                                                      vols_ttms=self.ivs_call_ttms)
        return vegas_ttms

    def reduce_ttms(self, ttms_ids: List[str]):
        if not np.all(np.isin(ttms_ids, self.ttms_ids)):
            raise ValueError(f"Expiries to be removed not present if chain")
        idx_ttms = np.where(np.isin(self.ttms_ids, ttms_ids))[0]
        strikes = [self.strikes_ttms[idx_ttm] for idx_ttm in idx_ttms]
        ivs_call_ttms = [self.ivs_call_ttms[idx_ttm] for idx_ttm in idx_ttms]
        ivs_put_ttms = [self.ivs_put_ttms[idx_ttm] for idx_ttm in idx_ttms]
        forwards = self.forwards[idx_ttms]

        assert self.call_oi is None and self.put_oi is None
        assert self.call_vol is None and self.put_vol is None

        chain = FutOptionChain(ccy=self.ccy,
                               ttms=self.ttms[idx_ttms],
                               forwards=forwards,
                               strikes_ttms=strikes,
                               ttms_ids=ttms_ids,
                               ivs_put_ttms=ivs_put_ttms,
                               ivs_call_ttms=ivs_call_ttms,
                               ticker=self.ticker)
        return chain