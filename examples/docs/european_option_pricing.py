"""Canonical script of docs/european_option_pricing.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here. The Fourier valuation of Section 5.1 of Sepp and Rakhmonov (2023) is
checked against Black-Scholes at zero vol-of-vol, against put-call parity, and for convergence in
the size of the integration grid. Select a case in ``Locals`` or run the file to execute every case.
"""
from enum import Enum

import numpy as np

import stochvolmodels as svm
from stochvolmodels.pricers.logsv_pricer import set_vol_scaler

QUICKSTART_PARAMS = svm.LogSvParams(sigma0=1.0, theta=1.0, kappa1=5.0, kappa2=5.0, beta=0.2,
                                    volvol=2.0)
# zero vol-of-vol and sigma0 = theta: volatility stays at 40% and prices are Black-Scholes prices
FLAT_PARAMS = svm.LogSvParams(sigma0=0.4, theta=0.4, kappa1=2.0, kappa2=2.0, beta=0.0, volvol=0.0)


def first_price() -> tuple:
    """Three-month at-the-money call of the quickstart: price and implied volatility."""
    return svm.LogSVPricer().price_vanilla(params=QUICKSTART_PARAMS, ttm=0.25, forward=1.0,
                                           strike=1.0, optiontype="C")


def fourier_prices(params: svm.LogSvParams, ttm: float, forward: float, strikes: np.ndarray,
                   optiontypes: np.ndarray, discfactor: float = 1.0,
                   n_points: int = 1000) -> np.ndarray:
    """Eqs. (5.4) and (5.9) on the grid Phi = -1/2 + iy of n_points points, as in price_slice."""
    vol_scaler = set_vol_scaler(sigma0=params.sigma0, ttm=ttm)
    phi_grid = svm.get_phi_grid(max_phi=n_points, vol_scaler=vol_scaler)
    zeros = np.zeros_like(phi_grid)
    _, log_mgf = svm.compute_logsv_a_mgf_grid(ttm=ttm, phi_grid=phi_grid, psi_grid=zeros,
                                              theta_grid=zeros, **params.to_dict())
    return svm.vanilla_slice_pricer_with_mgf_grid(log_mgf_grid=log_mgf, phi_grid=phi_grid,
                                                  forward=forward, strikes=strikes,
                                                  optiontypes=optiontypes, discfactor=discfactor)


def strike_grid(ttm: float, forward: float = 1.0, vol: float = 0.4, n: int = 13) -> tuple:
    """Strikes within three standard deviations, with puts below the forward and calls above."""
    deviation = vol * np.sqrt(ttm)
    strikes = forward * np.exp(np.linspace(-3.0 * deviation, 3.0 * deviation, n))
    return strikes, np.where(strikes >= forward, "C", "P")


def black_scholes_error(ttm: float, n_points: int = 1000) -> float:
    """Largest price error of the Fourier pricer against Black-Scholes at 40% volatility."""
    strikes, optiontypes = strike_grid(ttm)
    fourier = fourier_prices(FLAT_PARAMS, ttm=ttm, forward=1.0, strikes=strikes,
                             optiontypes=optiontypes, n_points=n_points)
    exact = svm.compute_bsm_vanilla_slice_prices(ttm=ttm, forward=1.0, strikes=strikes,
                                                 vols=np.full(strikes.shape, 0.4),
                                                 optiontypes=optiontypes)
    return np.max(np.abs(fourier - exact))


def analytic_prices() -> dict:
    """Black-Scholes price and implied volatility, and a Bachelier price with absolute vol."""
    call = svm.compute_bsm_vanilla_price(forward=100.0, strike=105.0, ttm=0.5, vol=0.2,
                                         optiontype="C", discfactor=0.98)
    vol = svm.infer_bsm_implied_vol(forward=100.0, ttm=0.5, strike=105.0, given_price=call,
                                    discfactor=0.98, optiontype="C")
    normal = svm.compute_normal_price(forward=100.0, strike=105.0, ttm=0.5, vol=20.0,
                                      discfactor=0.98, optiontype="C")
    return {"call": call, "implied_vol": vol, "normal_call": normal}


def low_volatility_error(vol: float, ttm: float, scale: float = 1.0) -> float:
    """Largest implied-vol error against Black-Scholes when sigma0 sqrt(ttm) is small.

    ``scale`` multiplies the default grid scale, which ``vol_scaler`` passes to the pricer.
    """
    params = svm.LogSvParams(sigma0=vol, theta=vol, kappa1=2.0, kappa2=2.0, beta=0.0, volvol=0.0)
    strikes, optiontypes = strike_grid(ttm, vol=vol)
    _, ivols = svm.LogSVPricer().price_slice(
        params=params, ttm=ttm, forward=1.0, strikes=strikes, optiontypes=optiontypes,
        vol_scaler=scale * set_vol_scaler(sigma0=vol, ttm=ttm))
    return np.nanmax(np.abs(ivols - vol))


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    FIRST_PRICE = 1
    SAME_AS_PRICE_SLICE = 2
    PUT_CALL_PARITY = 3
    BLACK_SCHOLES_LIMIT = 4
    GRID_CONVERGENCE = 5
    ANALYTIC = 6
    LOW_VOLATILITY = 7


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    if local == Locals.FIRST_PRICE:
        price, ivol = first_price()
        np.testing.assert_allclose([price, ivol], [0.197331, 0.999577], atol=1e-6)

    elif local == Locals.SAME_AS_PRICE_SLICE:
        strikes, optiontypes = strike_grid(0.25, vol=1.0)
        prices, _ = svm.LogSVPricer().price_slice(params=QUICKSTART_PARAMS, ttm=0.25, forward=1.0,
                                                  strikes=strikes, optiontypes=optiontypes)
        fourier = fourier_prices(QUICKSTART_PARAMS, 0.25, 1.0, strikes, optiontypes)
        np.testing.assert_allclose(fourier, prices, rtol=0.0, atol=1e-14)

    elif local == Locals.PUT_CALL_PARITY:
        strikes = np.linspace(0.5, 2.0, 16)
        calls = fourier_prices(QUICKSTART_PARAMS, 0.5, 1.0, strikes, np.full(16, "C"),
                               discfactor=0.97)
        puts = fourier_prices(QUICKSTART_PARAMS, 0.5, 1.0, strikes, np.full(16, "P"),
                              discfactor=0.97)
        # both are computed from the same capped payoff, so parity holds to rounding
        np.testing.assert_allclose(calls - puts, 0.97 * (1.0 - strikes), rtol=0.0, atol=1e-14)

    elif local == Locals.BLACK_SCHOLES_LIMIT:
        ttms = (1.0 / 52.0, 1.0 / 12.0, 1.0)
        errors = [black_scholes_error(ttm) for ttm in ttms]
        print(errors)
        assert 1e-7 < errors[0] < 2e-7 and max(errors[1:]) < 2e-10
        # the error follows exp(-pi / (2 dy)) of the spacing dy: aliasing near the pole at y = i/2
        spacings = [5.6 / set_vol_scaler(sigma0=0.4, ttm=ttm) / 999 for ttm in ttms]
        np.testing.assert_allclose(spacings, [0.101, 0.0687, 0.0687], atol=5e-4)
        ratios = np.array(errors) / np.exp(-np.pi / (2.0 * np.array(spacings)))
        assert np.all((0.5 < ratios) & (ratios < 2.0)), ratios

    elif local == Locals.GRID_CONVERGENCE:
        # with 200 and 400 points the grid is too coarse; the default of 1,000 is not
        for ttm in (1.0 / 52.0, 1.0 / 12.0, 1.0):
            coarse, medium = black_scholes_error(ttm, 200), black_scholes_error(ttm, 400)
            print(ttm, coarse, medium)
            assert 5e-3 < coarse < 3e-2 and 5e-5 < medium < 2e-3

    elif local == Locals.ANALYTIC:
        values = analytic_prices()
        np.testing.assert_allclose(values["call"], 3.5456, atol=1e-4)
        np.testing.assert_allclose(values["implied_vol"], 0.2, atol=1e-12)
        np.testing.assert_allclose(values["normal_call"], 3.4211, atol=1e-4)

    elif local == Locals.LOW_VOLATILITY:
        # the grid spacing grows as sigma0 sqrt(ttm) falls; 1,000 points are then too few
        np.testing.assert_allclose(5.6 / set_vol_scaler(sigma0=0.2, ttm=1.0 / 52.0) / 999, 0.202,
                                   atol=1e-3)
        errors = {(vol, ttm): low_volatility_error(vol, ttm)
                  for vol, ttm in ((0.2, 1.0 / 52.0), (0.1, 1.0 / 52.0), (0.1, 1.0 / 12.0))}
        print(errors)
        np.testing.assert_allclose(list(errors.values()), [0.091, 0.425, 0.112], atol=2e-3)
        # a grid scale twice the default helps but does not remove the error
        np.testing.assert_allclose(low_volatility_error(0.2, 1.0 / 52.0, scale=2.0), 0.0086,
                                   atol=5e-4)


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
