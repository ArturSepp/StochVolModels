"""Price a single maturity with the inverse-gamma normal-mixture adapter."""

from enum import Enum

import numpy as np

from stochvolmodels.data.option_chain import OptionSlice
from stochvolmodels.models.inverse_gamma import (
    InverseGammaNormalParams,
    InverseGammaNormalTerminalModel,
)
from stochvolmodels.pricers.inverse_gamma_pricer import InverseGammaNormalPricer


class Locals(Enum):
    SINGLE_MATURITY = 1


def run_local(local: Locals) -> None:
    """Price a single maturity with the inverse-gamma normal-mixture adapter."""
    if local != Locals.SINGLE_MATURITY:
        raise ValueError(local)
    params = InverseGammaNormalParams(alpha=4.0, beta=0.12, c=1.0, q=0.0, ttm=0.75)
    strikes = np.array([90.0, 100.0, 110.0])
    types = np.array(["P", "C", "C"])
    prices, vols = InverseGammaNormalPricer().price_slice(
        params=params, ttm=params.ttm, forward=100.0, strikes=strikes, optiontypes=types
    )
    option_slice = OptionSlice(
        ttm=params.ttm, forward=100.0, discfactor=1.0, id="9m", strikes=strikes, optiontypes=types
    )
    terminal = InverseGammaNormalTerminalModel(params)
    np.testing.assert_allclose(prices, terminal.price_european(option_slice))
    np.testing.assert_allclose(vols, terminal.implied_vols(option_slice), atol=1e-9)
    print("Strikes", strikes, "prices in currency units", prices, "annual Black IV (%)", 100 * vols)


if __name__ == "__main__":
    run_local(Locals.SINGLE_MATURITY)
