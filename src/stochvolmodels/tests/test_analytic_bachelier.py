import numpy as np
from scipy.stats import norm

from stochvolmodels.pricers.analytic import bachelier


def _reference_price(
    forward: float,
    strike: float,
    ttm: float,
    vol: float,
    optiontype: str,
    discfactor: float,
) -> float:
    standard_deviation = vol * np.sqrt(ttm)
    d = (forward - strike) / standard_deviation
    call = discfactor * (
        (forward - strike) * norm.cdf(d) + standard_deviation * norm.pdf(d)
    )
    if optiontype == "C":
        return call
    return call - discfactor * (forward - strike)


def test_bachelier_prices_match_reference_parity_and_slice() -> None:
    forward = 1.0
    ttm = 1.25
    discfactor = 0.96
    strikes = np.array([0.96, 1.0, 1.04])
    vols = np.array([0.025, 0.03, 0.04])
    optiontypes = np.array(["P", "C", "C"])

    actual = bachelier.compute_normal_slice_prices(
        ttm=ttm,
        forward=forward,
        strikes=strikes,
        vols=vols,
        optiontypes=optiontypes,
        discfactor=discfactor,
    )
    expected = np.array(
        [
            _reference_price(forward, strike, ttm, vol, optiontype, discfactor)
            for strike, vol, optiontype in zip(strikes, vols, optiontypes)
        ]
    )
    scalar = np.array(
        [
            bachelier.compute_normal_price(
                forward=forward,
                strike=strike,
                ttm=ttm,
                vol=vol,
                optiontype=optiontype,
                discfactor=discfactor,
            )
            for strike, vol, optiontype in zip(strikes, vols, optiontypes)
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-8)
    np.testing.assert_allclose(actual, scalar, rtol=0.0, atol=1.0e-14)

    strike = 1.02
    vol = 0.035
    call = bachelier.compute_normal_price(
        forward, strike, ttm, vol, discfactor, "C"
    )
    put = bachelier.compute_normal_price(
        forward, strike, ttm, vol, discfactor, "P"
    )
    np.testing.assert_allclose(call - put, discfactor * (forward - strike), atol=2.0e-8)


def test_bachelier_implied_vol_round_trip() -> None:
    forward = 1.0
    strike = 1.025
    ttm = 0.9
    discfactor = 0.98
    vol = 0.045
    price = bachelier.compute_normal_price(
        forward=forward,
        strike=strike,
        ttm=ttm,
        vol=vol,
        optiontype="P",
        discfactor=discfactor,
    )

    inferred = bachelier.infer_normal_implied_vol(
        forward=forward,
        strike=strike,
        ttm=ttm,
        given_price=price,
        optiontype="P",
        discfactor=discfactor,
    )

    np.testing.assert_allclose(inferred, vol, rtol=0.0, atol=2.0e-10)


def test_bachelier_delta_and_vega_match_finite_differences() -> None:
    forward = 1.0
    strike = 1.01
    ttm = 0.7
    vol = 0.04
    step = 1.0e-6

    price_up = bachelier.compute_normal_price(forward + step, strike, ttm, vol, 1.0, "C")
    price_down = bachelier.compute_normal_price(
        forward - step, strike, ttm, vol, 1.0, "C"
    )
    finite_delta = (price_up - price_down) / (2.0 * step)
    analytic_delta = bachelier.compute_normal_delta(ttm, forward, strike, vol, "C")

    vol_up = bachelier.compute_normal_price(forward, strike, ttm, vol + step, 1.0, "C")
    vol_down = bachelier.compute_normal_price(forward, strike, ttm, vol - step, 1.0, "C")
    finite_vega = (vol_up - vol_down) / (2.0 * step)
    analytic_vega = bachelier.compute_normal_slice_vegas(
        ttm=ttm,
        forward=forward,
        strikes=np.array([strike]),
        vols=np.array([vol]),
    )[0]

    np.testing.assert_allclose(analytic_delta, finite_delta, rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(analytic_vega, finite_vega, rtol=0.0, atol=2.0e-7)
