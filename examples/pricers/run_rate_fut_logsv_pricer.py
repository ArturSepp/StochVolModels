"""Price a SOFR futures option at the convexity-adjusted model forward."""

from enum import Enum

import numpy as np

from examples.docs.factor_hjm_stochastic_volatility import model_params
from stochvolmodels import ExpansionOrder
from stochvolmodels.data.option_chain import FutOptionChain
from stochvolmodels.pricers.factor_hjm.rate_logsv_pricer import (
    FutSettleType,
    RateFutLogSVPricer,
    calc_futures_rate,
)
from stochvolmodels.utils.rate_core import generate_ttms_grid, get_futures_start_and_pmt


class Locals(Enum):
    SOFR_OPTION = 1


def run_local(local: Locals) -> None:
    """Price a SOFR futures option at the convexity-adjusted model forward."""
    if local != Locals.SOFR_OPTION:
        raise ValueError(local)
    params = model_params()
    params.q = params.theta  # Expand around long-run volatility, as in the SOFR benchmark.
    ttms = np.array([1.0])
    start, end = get_futures_start_and_pmt(t0=ttms[0], lag=0.0, libor_tenor=0.25)
    forward = calc_futures_rate(
        ccy=params.ccy,
        basis_type="NELSON-SIEGEL",
        params=params,
        x0=np.zeros(params.basis.get_nb_factors()),
        y0=np.zeros(params.basis.get_nb_aux_factors()),
        sigma0=np.ones((1, 1)),
        t0=0.0,
        t_start=start,
        t_end=end,
        Delta=0.25,
        expansion_order=ExpansionOrder.ZERO,
        settlement_type=FutSettleType.SOFR,
    )[0][0]
    chain = FutOptionChain(
        ccy=params.ccy,
        ttms=ttms,
        forwards=np.array([forward]),
        strikes_ttms=[np.array([forward])],
        ttms_ids=np.array(["1y"]),
        ivs_call_ttms=[np.array([0.01])],
        ivs_put_ttms=[np.array([0.01])],
    )
    vols = RateFutLogSVPricer().price_chain(
        chain, params, t_grid=generate_ttms_grid(ttms), idxs=slice(0, 1)
    )
    iv = vols[0][0][0]
    assert 0 < iv < np.inf
    print(f"SOFR futures rate {forward:.4%}, ATM absolute normal IV {iv * 10000:.3f} bp/sqrt(year)")


if __name__ == "__main__":
    run_local(Locals.SOFR_OPTION)
