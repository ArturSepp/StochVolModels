"""Fit synthetic six-month smiles and quote a one-year, three-equity USD BRC.

Run this file against the installed core wheel. Prices are per 100 USD face.
The curves and quanto inputs below are illustrative.
"""

from dataclasses import replace

import numpy as np

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.fitters.sparse_local_vol import fit_sparse_local_vol
from stochvolmodels.models.basket import CorrelatedLocalVolModel
from stochvolmodels.models.lsv.simulation import LSVMonteCarloConfig
from stochvolmodels.pricers.barrier_reverse_convertible import (
    BrcConfig,
    BrcRequest,
    quote_brc_coupon,
    value_brc_batch,
)
from stochvolmodels.products.barrier_reverse_convertible import (
    BrcState,
    BrcTerms,
    Monitoring,
    TouchState,
)


def synthetic_market():
    """Fit the three illustrative native smiles once for the valuation snapshot."""
    moneyness = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    assets = ("SPX", "SX5E", "NKY")
    spots = np.array([5000.0, 4000.0, 30000.0])
    carry = np.array([0.03, 0.01, 0.0])
    tenor = 180 / 365
    surfaces = []
    for i, name in enumerate(assets):
        smile = 0.20 + i * 0.02 - 0.07 * np.log(moneyness)
        chain = OptionChain(
            ttms=np.array([tenor]),
            forwards=np.array([spots[i] * np.exp(carry[i] * tenor)]),
            discfactors=np.array([np.exp(-0.03 * tenor)]),
            strikes_ttms=[spots[i] * moneyness],
            optiontypes_ttms=[np.where(moneyness < 1, "P", "C")],
            ids=np.array([name]),
            mid_ivs=[smile],
        )
        fit = fit_sparse_local_vol(chain, spot=spots[i])
        if fit.status != "ok":
            raise ArithmeticError(f"{name} marginal failed: {fit.status}")
        surfaces.append(fit.surface)
    return CorrelatedLocalVolModel(
        asof="synthetic-issue",
        asset_ids=assets,
        currencies=("USD", "EUR", "JPY"),
        settlement_currency="USD",
        surfaces=tuple(surfaces),
        curve_times=[0, 1],
        forward_levels=[spots, spots * np.exp(carry)],
        correlation=[[1, 0.7, 0.6], [0.7, 1, 0.65], [0.6, 0.65, 1]],
        fx_vols=[0, 0.09, 0.12],
        equity_fx_correlations=[0, -0.2, 0.15],
    )


def main():
    """Quote issue coupons and revalue the issued coupons on a single path stream."""
    model = synthetic_market()
    assets, spots = model.asset_ids, model.spots
    cfg = BrcConfig(mc=LSVMonteCarloConfig(paths=8192, steps_per_year=252, seed=481))
    state = BrcState(asof=model.asof, touch=TouchState.CLEAR)
    discount = float(np.exp(-0.035))
    original = BrcTerms(
        asset_ids=assets,
        fixings=spots,
        barriers=0.7 * spots,
        fixing_time=1.0,
        payment_time=1.0,
        face=100.0,
        currency="USD",
    )
    requests = []
    print("barrier  monitoring                coupon %   MC SE (bp)   loss bounds %     status")
    for level in (0.6, 0.7, 0.8):
        for mode in Monitoring:
            terms = replace(original, barriers=level * spots, monitoring=mode)
            quote = quote_brc_coupon(
                terms, model, discount_factor=discount, discount_currency="USD", config=cfg
            )
            value = quote.valuation
            print(
                f"{level:6.0%}   {mode.value:23} {100 * quote.coupon:8.3f}   "
                f"{1e4 * quote.standard_error:8.2f}   "
                f"{100 * value.loss_bounds[0]:6.3f}..{100 * value.loss_bounds[1]:6.3f}    "
                f"{value.status}"
            )
            requests.append(
                BrcRequest(
                    terms=terms,
                    state=state,
                    coupon=quote.coupon,
                    discount_factor=discount,
                    discount_currency="USD",
                )
            )
    marks = value_brc_batch(requests, model, config=cfg)
    np.testing.assert_allclose([v.value for v in marks], 100.0, atol=1e-10, rtol=0)
    print("All six fixed-coupon marks reconcile to USD 100 at issuance.")


if __name__ == "__main__":
    main()
