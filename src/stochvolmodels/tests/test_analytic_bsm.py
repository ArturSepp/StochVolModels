import numpy as np
from scipy.stats import norm

import stochvolmodels as bsm


def _reference_price(
    forward: float,
    strike: float,
    ttm: float,
    vol: float,
    optiontype: str,
    discfactor: float,
) -> float:
    total_vol = vol * np.sqrt(ttm)
    d1 = np.log(forward / strike) / total_vol + 0.5 * total_vol
    d2 = d1 - total_vol
    if optiontype == "C":
        return discfactor * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return discfactor * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def test_bsm_prices_match_reference_parity_and_slice() -> None:
    forward = 1.03
    ttm = 0.75
    discfactor = 0.97
    strikes = np.array([0.9, 1.0, 1.1])
    vols = np.array([0.22, 0.25, 0.28])
    optiontypes = np.array(["P", "C", "C"])

    actual = bsm.compute_bsm_vanilla_slice_prices(
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
            bsm.compute_bsm_vanilla_price(
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

    # The package's Numba-compatible CDF is documented to be accurate to about 1.2e-7.
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(actual, scalar, rtol=0.0, atol=1.0e-14)

    strike = 1.07
    vol = 0.31
    call = bsm.compute_bsm_vanilla_price(
        forward, strike, ttm, vol, "C", discfactor
    )
    put = bsm.compute_bsm_vanilla_price(
        forward, strike, ttm, vol, "P", discfactor
    )
    np.testing.assert_allclose(call - put, discfactor * (forward - strike), atol=2.0e-8)


def test_bsm_implied_vol_round_trip() -> None:
    forward = 0.98
    strike = 1.04
    ttm = 1.4
    discfactor = 0.94
    vol = 0.37
    price = bsm.compute_bsm_vanilla_price(
        forward=forward,
        strike=strike,
        ttm=ttm,
        vol=vol,
        optiontype="P",
        discfactor=discfactor,
    )

    inferred = bsm.infer_bsm_implied_vol(
        forward=forward,
        strike=strike,
        ttm=ttm,
        given_price=price,
        optiontype="P",
        discfactor=discfactor,
    )

    np.testing.assert_allclose(inferred, vol, rtol=0.0, atol=2.0e-12)


def test_bsm_delta_and_vega_match_finite_differences() -> None:
    forward = 1.02
    strike = 0.99
    ttm = 0.8
    vol = 0.29
    step = 1.0e-5

    price_up = bsm.compute_bsm_vanilla_price(forward + step, strike, ttm, vol, "C")
    price_down = bsm.compute_bsm_vanilla_price(forward - step, strike, ttm, vol, "C")
    finite_delta = (price_up - price_down) / (2.0 * step)
    analytic_delta = bsm.compute_bsm_vanilla_delta(ttm, forward, strike, vol, "C")

    vol_up = bsm.compute_bsm_vanilla_price(forward, strike, ttm, vol + step, "C")
    vol_down = bsm.compute_bsm_vanilla_price(forward, strike, ttm, vol - step, "C")
    finite_vega = (vol_up - vol_down) / (2.0 * step)
    analytic_vega = bsm.compute_bsm_vanilla_vega(ttm, forward, strike, vol)

    total_vol = vol * np.sqrt(ttm)
    d1 = np.log(forward / strike) / total_vol + 0.5 * total_vol
    reference_delta = norm.cdf(d1)
    reference_vega = forward * norm.pdf(d1) * np.sqrt(ttm)

    np.testing.assert_allclose(analytic_delta, reference_delta, rtol=0.0, atol=5.0e-8)
    np.testing.assert_allclose(analytic_vega, reference_vega, rtol=0.0, atol=1.0e-14)
    # Differentiation amplifies the small approximation error in the package CDF.
    np.testing.assert_allclose(analytic_delta, finite_delta, rtol=0.0, atol=1.0e-5)
    np.testing.assert_allclose(analytic_vega, finite_vega, rtol=0.0, atol=1.0e-5)


def test_bsm_theta_matches_negative_ttm_derivative() -> None:
    forward = 1.01
    strike = 1.05
    ttm = 0.6
    vol = 0.24
    step = 1.0e-5

    longer = bsm.compute_bsm_vanilla_price(forward, strike, ttm + step, vol, "P")
    shorter = bsm.compute_bsm_vanilla_price(forward, strike, ttm - step, vol, "P")
    finite_theta = -(longer - shorter) / (2.0 * step)
    analytic_theta = bsm.compute_bsm_vanilla_theta(ttm, forward, strike, vol, "P")

    np.testing.assert_allclose(analytic_theta, finite_theta, rtol=0.0, atol=2.0e-7)
