"""Compare ATM calls conditional on the growth and stress regimes."""

from enum import Enum

import numpy as np

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.regime_logsv import (
    CrraRiskPremia,
    Regime,
    RegimeLogSvDynamics,
    RegimeSwitchLogSvParams,
    RegimeTransition,
)
from stochvolmodels.pricers.regime_switch_logsv_pricer import RegimeSwitchLogSVPricer


class Locals(Enum):
    CONDITIONAL_CALLS = 1


def run_local(local: Locals) -> None:
    """Compare ATM calls conditional on the growth and stress regimes."""
    if local != Locals.CONDITIONAL_CALLS:
        raise ValueError(local)
    common = dict(kappa1=2.6949, kappa2=10.0107, beta=-1.5082, volvol=0.8503)
    params = RegimeSwitchLogSvParams(
        sigma0=0.15,
        regimes=(
            RegimeLogSvDynamics(theta=0.15, **common),
            RegimeLogSvDynamics(theta=0.225, **common),
        ),
        transitions=(RegimeTransition(0.10, -0.25 / 0.75), RegimeTransition(1.0, 0.15 / 1.15)),
        risk_premia=CrraRiskPremia(utility_power=-0.5, agent_horizon=3.0),
    )
    chain = OptionChain.get_uniform_chain(
        np.array([0.25]), np.array(["3m"]), strikes=np.array([1.0])
    )
    result = RegimeSwitchLogSVPricer().compute_state_conditional_prices_with_vols(
        chain, params, max_phi=401
    )
    for regime in Regime:
        prices, vols = result.for_regime(regime)
        assert prices[0][0] > 0 and 0 < vols[0][0] < np.inf
        print(
            f"{regime.name} ATM price {prices[0][0]:.6f} per unit forward, annual Black IV {vols[0][0]:.3%}"
        )


if __name__ == "__main__":
    run_local(Locals.CONDITIONAL_CALLS)
