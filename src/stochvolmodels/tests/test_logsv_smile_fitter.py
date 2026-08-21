import numpy as np
import pandas as pd
import vanilla_option_pricers as bsm

from stochvolmodels.data.sample_option_chains import get_oca_simulated_chain_data
from stochvolmodels.fitters import (
    ATM_VOL,
    BETA,
    VOLVOL,
    calc_logsv_ivols,
    fit_logsv_ivols,
    generate_grid_option_prices_from_slice,
)


def test_fit_logsv_ivols_recovers_synthetic_smile() -> None:
    log_strikes = np.linspace(-0.25, 0.25, 31)
    expected = {ATM_VOL: 0.22, BETA: -0.35, VOLVOL: 0.60}
    mid_vols = calc_logsv_ivols(log_strikes=log_strikes, **expected)

    fitted = fit_logsv_ivols(
        log_strikes=log_strikes,
        mid_vols=mid_vols,
        ttm=30.0 / 365.0,
    )

    np.testing.assert_allclose(
        [fitted[ATM_VOL], fitted[BETA], fitted[VOLVOL]],
        [expected[ATM_VOL], expected[BETA], expected[VOLVOL]],
        rtol=2e-5,
        atol=2e-6,
    )


def test_generated_grid_prices_use_vanilla_option_pricers() -> None:
    forward = 100.0
    ttm = 30.0 / 365.0
    given_log_strikes = np.linspace(-0.2, 0.2, 21)
    vols = pd.Series(
        calc_logsv_ivols(given_log_strikes, sigma0=0.25, beta=-0.2, volvol=0.5)
    )
    grid = np.linspace(-0.15, 0.15, 17)

    puts, calls = generate_grid_option_prices_from_slice(
        vols=vols,
        given_log_strikes=given_log_strikes,
        log_strike_grid=grid,
        p0_ref=forward,
        ttm=ttm,
    )

    strikes = forward * np.exp(grid)
    np.testing.assert_allclose(calls.to_numpy() - puts.to_numpy(), forward - strikes)
    direct_calls = bsm.compute_bsm_vanilla_slice_prices(
        ttm=ttm,
        forward=forward,
        strikes=strikes,
        vols=calc_logsv_ivols(grid, **fit_logsv_ivols(given_log_strikes, vols, ttm)),
        optiontypes=np.full(strikes.shape, 'C'),
    )
    np.testing.assert_allclose(calls.to_numpy(), direct_calls)


def test_bundled_oca_chain_supports_smile_fitting() -> None:
    option_chain = get_oca_simulated_chain_data()

    assert option_chain.ticker == "OCA_SIM"
    assert option_chain.ids.tolist() == ["1w: 12Jan2024", "1m: 16Feb2024"]
    assert len(option_chain.ttms) == 2

    idx = 1
    log_strikes = np.log(option_chain.strikes_ttms[idx] / option_chain.forwards[idx])
    mid_vols = 0.5 * (option_chain.bid_ivs[idx] + option_chain.ask_ivs[idx])
    fitted = fit_logsv_ivols(log_strikes, mid_vols, option_chain.ttms[idx])
    fitted_vols = calc_logsv_ivols(log_strikes, **fitted)

    assert np.sqrt(np.mean(np.square(fitted_vols - mid_vols))) < 1e-6
