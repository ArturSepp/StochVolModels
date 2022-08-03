"""
contained for option chain data
"""

from __future__ import annotations

# built in
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from numba.typed import List

# analytics
import utils.bsm_pricer as bsm


@dataclass
class ChainData:
    """
    container for chain data
    """
    ttms: np.ndarray
    forwards: np.ndarray
    strikes_ttms: Tuple[np.ndarray, ...]
    optiontypes_ttms: Tuple[np.ndarray, ...]
    discfactors: Optional[np.ndarray] = None  # discount factors
    discount_rates: Optional[np.ndarray] = None  # discount rates
    ids: Optional[np.ndarray] = None  # slice_t names
    ticker: Optional[str] = None  # associated ticker
    bid_ivs: Optional[Tuple[np.ndarray, ...]] = None
    ask_ivs: Optional[Tuple[np.ndarray, ...]] = None

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

    @classmethod
    def slice_to_chain(cls,
                       ttm: float,
                       forward: float,
                       strikes: np.ndarray,
                       optiontypes: np.ndarray,
                       discfactor: float = 1.0
                       ) -> ChainData:

        return cls(ttms=np.array([ttm]),
                   forwards=np.array([forward]),
                   strikes_ttms=(strikes,),
                   optiontypes_ttms=(optiontypes,),
                   discfactors=np.array([discfactor]))

    def get_mid_vols(self) -> List[np.ndarray]:
        return List(0.5 * (bid_iv + ask_iv) for bid_iv, ask_iv in zip(self.bid_ivs, self.ask_ivs))

    def get_chain_vegas(self, is_unit_ttm_vega: bool = False) -> List[np.ndarray]:
        if is_unit_ttm_vega:
            ttms = np.ones_like(self.ttms)
        else:
            ttms = self.ttms
        vegas_ttms = bsm.bsm_vegas_ttms(ttms=ttms,
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

    def get_chain_data_as_xy(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        these data are needed for to pass x and y for model calibrations
        """
        mid_vols = List(0.5 * (bid_iv + ask_iv) for bid_iv, ask_iv in zip(self.bid_ivs, self.ask_ivs))
        x = (self.ttms, self.forwards, self.discfactors, self.strikes_ttms, self.optiontypes_ttms)
        y = mid_vols
        return x, y

    def compute_model_ivols_from_chain_data(self, model_prices: List[np.ndarray]) -> List[np.ndarray]:
        model_ivols = bsm.model_chain_prices_to_bsm_ivols(ttms=self.ttms,
                                                          forwards=self.forwards,
                                                          discfactors=self.discfactors,
                                                          strikes_ttms=self.strikes_ttms,
                                                          optiontypes_ttms=self.optiontypes_ttms,
                                                          model_prices_ttms=model_prices)
        return model_ivols

    @classmethod
    def to_uniform_strikes(cls, obj, num_strikes=21):
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
