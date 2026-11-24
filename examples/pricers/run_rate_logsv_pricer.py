"""Price a USD swaption cube in absolute normal volatility units."""

from enum import Enum

import numpy as np

from examples.docs.factor_hjm_stochastic_volatility import (
    EXPIRY,
    TENORS,
    forward_swap_rates,
    model_params,
)
from stochvolmodels.data.option_chain import SwOptionChain
from stochvolmodels.pricers.factor_hjm.rate_logsv_pricer import RateLogSVPricer
from stochvolmodels.utils.rate_core import generate_ttms_grid


class Locals(Enum):
    ATM_SWAPTIONS = 1


def run_local(local: Locals) -> None:
    """Price a USD swaption cube in absolute normal volatility units."""
    if local != Locals.ATM_SWAPTIONS:
        raise ValueError(local)
    params = model_params()
    ttms = np.array([EXPIRY])
    forwards = [np.array([f]) for f in forward_swap_rates(params)]
    chain = SwOptionChain(
        ccy=params.ccy,
        ttms=ttms,
        tenors=TENORS,
        ttms_ids=["2y"],
        tenors_ids=["2y", "5y", "10y"],
        forwards=forwards,
        strikes_ttms=[[f.copy()] for f in forwards],
        bid_ivs=[[np.array([0.009])] for _ in TENORS],
        ask_ivs=[[np.array([0.011])] for _ in TENORS],
    )
    vols = RateLogSVPricer().price_chain(
        chain, params, t_grid=generate_ttms_grid(ttms), idxs=slice(0, 1)
    )
    for tenor, values in zip(TENORS, vols, strict=True):
        iv = values[0][0]
        assert 0 < iv < np.inf
        print(f"2y x {tenor:g}y ATM absolute normal IV {iv * 10000:.3f} bp/sqrt(year)")


if __name__ == "__main__":
    run_local(Locals.ATM_SWAPTIONS)
