"""Quote and mark USD accruals on a synthetic EUR equity from an installed core wheel."""

import numpy as np

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.fitters.sparse_local_vol import fit_sparse_local_vol
from stochvolmodels.models.basket import CorrelatedLocalVolModel
from stochvolmodels.pricers.accumulator import quote_accumulator_strike, value_accumulator
from stochvolmodels.products.accumulator import (
    AccumulatorDirection,
    AccumulatorState,
    AccumulatorTerms,
)


def main() -> None:
    """Fit a six-month smile, quote both directions and settle earned cash on a touch."""
    spot, carry, tenor = 100.0, 0.015, 180 / 365
    strikes = spot * np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    quotes = OptionChain(
        ttms=np.array([tenor]),
        forwards=np.array([spot * np.exp(carry * tenor)]),
        discfactors=np.array([np.exp(-0.02 * tenor)]),
        strikes_ttms=[strikes],
        optiontypes_ttms=[np.where(strikes < spot, "P", "C")],
        ids=np.array(["6M"]),
        mid_ivs=[np.array([0.25, 0.235, 0.22, 0.212, 0.204])],
    )
    fit = fit_sparse_local_vol(quotes, spot=spot)
    if fit.status != "ok":
        raise ArithmeticError(f"Marginal fit failed: {fit.status}")
    model = CorrelatedLocalVolModel(
        asof="issue",
        asset_ids=("EQ",),
        currencies=("EUR",),
        settlement_currency="USD",
        surfaces=(fit.surface,),
        curve_times=[0, 1],
        forward_levels=[[spot], [spot * np.exp(carry)]],
        correlation=[[1]],
        fx_vols=[0.12],
        equity_fx_correlations=[-0.20],
    )
    times = np.arange(1, 22) / 252
    for side in AccumulatorDirection:
        terms = AccumulatorTerms(
            asset_id="EQ",
            currency="USD",
            original_spot=spot,
            notional=100,
            direction=side,
            barrier=105 if side is AccumulatorDirection.ACCUMULATOR else 95,
            fixing_times=times,
            payment_indices=np.zeros(21, dtype=int),
            payment_times=[times[-1]],
        )
        args = dict(discount_factors=np.exp(-0.03 * terms.payment_times), discount_currency="USD")
        quote = quote_accumulator_strike(terms, model, **args)
        if quote.status != "ok":
            raise ArithmeticError(f"Quote unavailable: {quote.status}")
        state = AccumulatorState.at_issue(terms, model.asof)
        mark = value_accumulator(terms, state, model, strike_ratio=quote.strike_ratio, **args)
        assert mark.status == "ok" and abs(mark.value) < 0.001
        stopped, _ = state.advance(
            terms,
            asof="second close",
            elapsed=times[1],
            closes=[spot, terms.barrier],
            strike_ratio=quote.strike_ratio,
        )
        df = np.exp(-0.03 * (terms.payment_times - stopped.elapsed))
        fixed = value_accumulator(
            terms,
            stopped,
            None,
            strike_ratio=quote.strike_ratio,
            discount_factors=df,
            discount_currency="USD",
        )
        assert fixed.status == "fixed"
        np.testing.assert_allclose(fixed.value, df[0] * terms.fixing_cash(spot, quote.strike_ratio))
        print(
            f"{side.value}: strike {quote.strike_ratio:.5%} of original spot, "
            f"issue PV {mark.value:.7f} USD, stopped PV {fixed.value:.5f} USD"
        )


if __name__ == "__main__":
    main()
