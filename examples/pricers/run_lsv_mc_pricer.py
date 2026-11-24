"""Estimate a European call and its sampling error with antithetic LSV paths."""

from enum import Enum

import numpy as np
from scipy.special import ndtr

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.logsv.params import LogSvParams
from stochvolmodels.models.lsv import LSVModel
from stochvolmodels.models.lsv.simulation import LSVMonteCarloConfig
from stochvolmodels.pricers.lsv_mc_pricer import LSVMonteCarloPricer


class Locals(Enum):
    SEEDED_CALL = 1


def run_local(local: Locals) -> None:
    """Estimate a European call and its sampling error with antithetic LSV paths."""
    if local != Locals.SEEDED_CALL:
        raise ValueError(local)
    vol, ttm = 0.2, 0.5
    model = LSVModel.from_params(LogSvParams(sigma0=vol, theta=vol, beta=0.0, volvol=0.0))
    chain = OptionChain.get_uniform_chain(
        np.array([ttm]), np.array(["6m"]), strikes=np.array([1.0])
    )
    config = LSVMonteCarloConfig(paths=8192, steps_per_year=48, seed=7, antithetic=True)
    result = LSVMonteCarloPricer(config=config).price_chain_detailed(chain, model)
    price, se = result.prices[0][0], result.standard_errors[0][0]
    black = 2 * ndtr(vol * np.sqrt(ttm) / 2) - 1
    assert se > 0 and abs(price - black) < 4 * se
    print(f"ATM price {price:.6f} per unit forward, standard error {se:.6f}, Black {black:.6f}")


if __name__ == "__main__":
    run_local(Locals.SEEDED_CALL)
