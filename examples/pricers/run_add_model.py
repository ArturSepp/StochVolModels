"""Extend the chain pricer with a constant Black volatility and check its quotes."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.special import erf
from vanilla_option_pricers import compute_bsm_vanilla_slice_prices

from stochvolmodels import OptionChain, VariableType
from stochvolmodels.models.base import ModelParams
from stochvolmodels.pricers.model_pricer import ModelPricer


@dataclass
class ConstantVolParams(ModelParams):
    """Annualised standard deviation of log returns under the forward measure."""

    volatility: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.volatility) or self.volatility < 0:
            raise ValueError("Black volatility must be finite and nonnegative")


class ConstantVolPricer(ModelPricer):
    """Return discounted European premiums in the forward's currency units."""

    def price_chain(
        self, option_chain: OptionChain, params: ConstantVolParams,
        variable_type: VariableType = VariableType.LOG_RETURN, **kwargs,
    ) -> list[np.ndarray]:
        """Price an ordinary call/put chain, preserving its expiry and strike order."""
        if variable_type != VariableType.LOG_RETURN:
            raise NotImplementedError("This example prices European calls and puts on the asset")
        prices = []
        for ttm, forward, discount, strikes, optiontypes in zip(
            option_chain.ttms, option_chain.forwards, option_chain.discfactors,
            option_chain.strikes_ttms, option_chain.optiontypes_ttms, strict=True,
        ):
            if not np.isin(optiontypes, ("C", "P")).all():
                raise ValueError("This example supports ordinary call (C) and put (P) payoffs")
            prices.append(compute_bsm_vanilla_slice_prices(
                ttm=ttm, forward=forward, discfactor=discount, strikes=strikes,
                vols=np.full(len(strikes), params.volatility), optiontypes=optiontypes,
            ))
        return prices


class Locals(Enum):
    CALL_PUT_SLICE = 1


def run_local(local: Locals) -> None:
    """Check parity away from the forward and the independent ATM Black formula."""
    if local != Locals.CALL_PUT_SLICE:
        raise ValueError(local)
    params = ConstantVolParams(volatility=0.20)
    forward, ttm, discount = 100.0, 0.5, 0.97
    strikes = np.array([95.0, 95.0, forward])
    prices, ivols = ConstantVolPricer().price_slice(
        params=params, ttm=ttm, forward=forward, discfactor=discount,
        strikes=strikes, optiontypes=np.array(["C", "P", "C"]),
    )
    parity = discount * (forward - strikes[0])
    atm = discount * forward * erf(params.volatility * np.sqrt(ttm / 8.0))
    np.testing.assert_allclose(prices[0] - prices[1], parity, rtol=0, atol=1e-10)
    np.testing.assert_allclose(prices[2], atm, rtol=0, atol=1e-10)
    np.testing.assert_allclose(ivols, params.volatility, rtol=0, atol=1e-8)
    print(f"Call minus put {prices[0] - prices[1]:.8f} (parity {parity:.8f})")
    print(f"ATM call {prices[2]:.8f} (Black {atm:.8f}), annual Black IVs {ivols}")


if __name__ == "__main__":
    run_local(Locals.CALL_PUT_SLICE)
