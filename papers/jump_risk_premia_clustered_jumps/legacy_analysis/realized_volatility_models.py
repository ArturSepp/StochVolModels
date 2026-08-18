# built in
import pandas as pd
import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
from enum import Enum

# qis
import qis.utils.dates as da
import qis.perfstats.returns as ret
import qis.models.linear.ewm as ewm

# pricers
from stochvolmodels.pricers.analytic.bsm import compute_bsm_vanilla_price
from stochvolmodels.pricers.hawkes_jd_pricer import HawkesJDPricer

# test data
from sigma_strats.data.price_data import Frequency, load_data

from papers.jump_risk_premia_clustered_jumps import hawkes_estimator as haw


class RV_MODEL(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_expected_qv(self, value_time: pd.Timestamp) -> float:
        pass

    @abstractmethod
    def get_option_model_price(self, value_time: pd.Timestamp, ttm: float = 1.0, forward: float = 1.0,
                               optiontype: str = 'C', strike: float = 1.0
                               ) -> Tuple[float, float]:
        pass

    def get_option_slice_price(self, value_time: pd.Timestamp, strikes: np.ndarray, optiontypes: np.ndarray,
                               ttm: float = 1.0, forward: float = 1.0,
                               ) -> Tuple[np.ndarray, np.ndarray]:
        pass


class EWMA_RV_MODEL(RV_MODEL):

    def __init__(self, price: pd.Series,
                 af: float = 365.0,
                 ewm_lambda: float = 0.94
                 ):
        returns = ret.to_returns(prices=price, is_log_returns=True)
        vols = np.sqrt(af) * ewm.compute_ewm_vol(data=returns, ewm_lambda=ewm_lambda, annualize=False)
        self.model_vols = vols.shift(1)
        super().__init__()

    def get_expected_qv(self, value_time: pd.Timestamp) -> float:
        idx = self.model_vols.index.get_indexer([value_time], method='ffill')
        vol = self.model_vols.iloc[idx].to_numpy()[0]
        return vol

    def get_option_model_price(self,
                               value_time: pd.Timestamp,
                               ttm: float = 1.0,
                               forward: float = 1.0,
                               optiontype: str = 'C',
                               strike: float = 1.0
                               ) -> Tuple[float, float]:
        vol = self.get_expected_qv(value_time=value_time)
        return compute_bsm_vanilla_price(forward=forward, strike=strike, ttm=ttm, vol=vol, optiontype=optiontype), vol


class HAWKES_RV_MODEL(RV_MODEL):

    def __init__(self, price: pd.Series,
                 af: float = 365.0,
                 mid_vol_span: float = 7,
                 vol_risk_premia: float = 0.0,
                 ):
        # returns = ret.to_returns(prices=price, is_log_returns=True, is_first_zero=True)
        self.model_params = haw.estimate_hawkes_jd_independent(price=price, af=af)
        #haw.illustrate_hawkes_jd(price=price, model_params=self.model_params, af=af)
        #plt.show()
        vol_hawks, model_data = haw.forecast_hawkes_jd_vol(price=price, model_params=self.model_params, mid_vol_span=mid_vol_span, af=365.0)
        self.vol_hawks = vol_hawks
        self.model_data = model_data.shift(1)  # for non-anticipation shift
        self.pricer = HawkesJDPricer()
        self.vol_risk_premia = vol_risk_premia
        super().__init__()

    def get_expected_qv(self, value_time: pd.Timestamp) -> float:
        idx = self.vol_hawks.index.get_indexer([value_time], method='ffill')
        vol = self.vol_hawks.iloc[idx].to_numpy()[0]
        return vol

    def get_option_model_price(self,
                               value_time: pd.Timestamp,
                               ttm: float = 1.0,
                               forward: float = 1.0,
                               optiontype: str = 'C',
                               strike: float = 1.0
                               ) -> Tuple[float, float]:
        idx = self.model_data.index.get_indexer([value_time], method='ffill')
        model_data_t = self.model_data.iloc[idx, :]
        model_params = self.model_params
        model_params.lambda_p, model_params.lambda_m, sigma = model_data_t['lambda_p'].iloc[0], model_data_t['lambda_m'].iloc[0], model_data_t['sigma'].iloc[0]
        model_params.lambda_p = 0.5*model_params.lambda_p
        model_params.lambda_m = 0.5*model_params.lambda_m
        model_params.theta_p = 0.5*model_params.theta_p
        model_params.theta_m = 0.5*model_params.theta_m
        model_params.sigma = np.maximum(sigma+self.vol_risk_premia, 0.1)
        model_price, vol = self.pricer.price_vanilla(params=model_params, ttm=ttm, forward=forward, strike=strike, optiontype=optiontype)
        return model_price, vol

    def get_option_slice_price(self,
                               value_time: pd.Timestamp,
                               strikes: np.ndarray,
                               optiontypes: np.ndarray,
                               ttm: float = 1.0,
                               forward: float = 1.0,
                               ) -> Tuple[np.ndarray, np.ndarray]:
        idx = self.model_data.index.get_indexer([value_time], method='ffill')
        model_data_t = self.model_data.iloc[idx, :]
        model_params = self.model_params
        model_params.lambda_p, model_params.lambda_m, model_params.sigma = model_data_t['lambda_p'].iloc[0], model_data_t['lambda_m'].iloc[0], model_data_t['sigma'].iloc[0]
        model_price, vol = self.pricer.price_slice(params=model_params, ttm=ttm, forward=forward, strikes=strikes, optiontypes=optiontypes)
        return model_price, vol


class LocalTests(Enum):
    EWMA_RV = 1
    ILLUSTRATE_HAWKES_VOLS = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    ticker = 'BTC'
    frequency = Frequency.DAILY
    time_period = da.TimePeriod(None, pd.Timestamp('2022-11-19'))

    price = load_data(ticker=ticker, time_period=time_period, frequency=frequency)

    if local_test == LocalTests.EWMA_RV:
        rv_model = EWMA_RV_MODEL(price=price)

        value_time = pd.Timestamp('2022-10-02 08:00:00+00:00')
        price = rv_model.get_option_model_price(value_time=value_time)
        print(price)


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.ILLUSTRATE_HAWKES_VOLS)
