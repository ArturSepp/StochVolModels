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
from numba.typed import List

import stochvolmodels.pricers.core.bsm_pricer as bsm


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
                    ids=self.ids)
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
        deltas_ttms = bsm.compute_bsm_deltas_ttms(ttms=self.ttms,
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

    @classmethod
    def get_slices_as_chain(cls, option_chain: OptionChain, ids: List[str]) -> OptionChain:
        """
        return a subset of chain for given ids
        """
        if len(ids) == 1:
            idx = option_chain.ids.tolist().index(ids[0])
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
            indices = np.in1d(option_chain.ids, ids).nonzero()[0]
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