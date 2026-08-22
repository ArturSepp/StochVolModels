"""Independent contracts for the stable Gaussian-mixture and Student-t pricers."""

from types import SimpleNamespace

import numpy as np
import pytest
from numba.typed import List

import stochvolmodels.pricers.gmm_pricer as gmm_module
import stochvolmodels.pricers.tdist_pricer as tdist_module
import stochvolmodels.fitters.tdist as tdist_fitter
from stochvolmodels import (
    CalibrationError,
    GmmParams,
    GmmPricer,
    OptionChain,
    TdistParams,
    TdistPricer,
    compute_bsm_vanilla_price,
)
from stochvolmodels.fitters.tdist import imply_drift_tdist
from stochvolmodels.pricers.gmm_pricer import (
    compute_gmm_vanilla_price,
    compute_gmm_vanilla_slice_prices,
    gmm_vanilla_chain_pricer,
)


def _quoted_slice() -> OptionChain:
    """Return a small deterministic one-maturity chain suitable for calibration tests."""
    return OptionChain(
        ttms=np.array([0.5]),
        forwards=np.array([1.0]),
        strikes_ttms=List([np.array([0.9, 1.0, 1.1])]),
        optiontypes_ttms=List([np.array(["P", "C", "C"])]),
        ids=np.array(["6m"]),
        discount_rates=np.array([0.02]),
        bid_ivs=List([np.array([0.24, 0.22, 0.23])]),
        ask_ivs=List([np.array([0.26, 0.24, 0.25])]),
    )


def _synthetic_tdist_slice(params: TdistParams) -> OptionChain:
    """Generate exact Student-t volatility quotes from the public pricing route."""
    chain = OptionChain.slice_to_chain(
        ttm=params.ttm,
        forward=1.0,
        strikes=np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
        optiontypes=np.array(["P", "P", "C", "C", "C"]),
        discfactor=np.exp(-0.02 * params.ttm),
        id="synthetic",
    )
    prices = TdistPricer().price_chain(chain, params)
    ivols = np.asarray(chain.compute_model_ivols_from_chain_data(prices)[0])
    return OptionChain(
        ttms=chain.ttms,
        forwards=chain.forwards,
        strikes_ttms=chain.strikes_ttms,
        optiontypes_ttms=chain.optiontypes_ttms,
        ids=chain.ids,
        discfactors=chain.discfactors,
        bid_ivs=List([ivols]),
        ask_ivs=List([ivols]),
    )


def _synthetic_gmm_slice(params: GmmParams) -> OptionChain:
    """Generate exact Gaussian-mixture volatility quotes from the public route."""
    chain = OptionChain.slice_to_chain(
        ttm=params.ttm,
        forward=1.0,
        strikes=np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
        optiontypes=np.array(["P", "P", "C", "C", "C"]),
        discfactor=np.exp(-0.02 * params.ttm),
        id="synthetic",
    )
    prices = GmmPricer().price_chain(chain, params)
    ivols = np.asarray(chain.compute_model_ivols_from_chain_data(prices)[0])
    return OptionChain(
        ttms=chain.ttms,
        forwards=chain.forwards,
        strikes_ttms=chain.strikes_ttms,
        optiontypes_ttms=chain.optiontypes_ttms,
        ids=chain.ids,
        discfactors=chain.discfactors,
        bid_ivs=List([ivols]),
        ask_ivs=List([ivols]),
    )


def test_one_state_gmm_reduces_to_black_scholes() -> None:
    """A martingale one-state Gaussian mixture is exactly Black--Scholes."""
    vol = 0.35
    ttm = 0.75
    forward = 1.05
    strike = 1.1
    discfactor = 0.97
    params = GmmParams(
        gmm_weights=np.array([1.0]),
        gmm_mus=np.array([-0.5 * vol**2]),
        gmm_vols=np.array([vol]),
        ttm=ttm,
    )

    actual, _ = GmmPricer().price_vanilla(
        params=params,
        ttm=ttm,
        forward=forward,
        strike=strike,
        optiontype="C",
        discfactor=discfactor,
    )
    expected = compute_bsm_vanilla_price(
        forward=forward,
        strike=strike,
        ttm=ttm,
        vol=vol,
        optiontype="C",
        discfactor=discfactor,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-14)
    np.testing.assert_allclose(
        compute_gmm_vanilla_price.py_func(
            gmm_weights=params.gmm_weights,
            gmm_mus=params.gmm_mus,
            gmm_vols=params.gmm_vols,
            ttm=ttm,
            forward=forward,
            strike=strike,
            optiontype="C",
            discfactor=discfactor,
        ),
        actual,
        rtol=0.0,
        atol=1.0e-14,
    )


def test_gmm_python_slice_and_chain_routes_match_compiled_pricer() -> None:
    params = GmmParams(
        gmm_weights=np.array([0.35, 0.65]),
        gmm_mus=np.array([-0.12, 0.02]),
        gmm_vols=np.array([0.22, 0.35]),
        ttm=0.5,
    )
    chain = OptionChain.slice_to_chain(
        ttm=params.ttm,
        forward=1.0,
        strikes=np.array([0.9, 1.0, 1.1]),
        optiontypes=np.array(["P", "C", "C"]),
        discfactor=0.99,
        id="6m",
    )
    compiled = np.asarray(GmmPricer().price_chain(chain, params)[0])
    python_slice = compute_gmm_vanilla_slice_prices.py_func(
        gmm_weights=params.gmm_weights,
        gmm_mus=params.gmm_mus,
        gmm_vols=params.gmm_vols,
        ttm=params.ttm,
        forward=chain.forwards[0],
        strikes=chain.strikes_ttms[0],
        optiontypes=chain.optiontypes_ttms[0],
        discfactor=chain.discfactors[0],
    )
    python_chain = gmm_vanilla_chain_pricer.py_func(
        gmm_weights=params.gmm_weights,
        gmm_mus=params.gmm_mus,
        gmm_vols=params.gmm_vols,
        ttms=chain.ttms,
        forwards=chain.forwards,
        strikes_ttms=chain.strikes_ttms,
        optiontypes_ttms=chain.optiontypes_ttms,
        discfactors=chain.discfactors,
    )

    np.testing.assert_allclose(python_slice, compiled, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(python_chain[0], compiled, rtol=0.0, atol=1.0e-14)


def test_gmm_parameter_density_and_sorting_contracts() -> None:
    params = GmmParams(
        gmm_weights=np.array([0.6, 0.4]),
        gmm_mus=np.array([0.05, -0.1]),
        gmm_vols=np.array([0.3, 0.2]),
        ttm=0.5,
    )
    grid = np.linspace(-2.0, 2.0, 20_001)
    state_pdfs, aggregate = params.compute_state_pdfs(grid)

    np.testing.assert_allclose(params.compute_pdf(grid), aggregate, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(np.trapezoid(aggregate, grid), 1.0, rtol=0.0, atol=1.0e-9)
    assert state_pdfs.shape == (grid.size, 2)
    np.testing.assert_allclose(
        params.get_get_avg_vol(),
        np.sqrt(np.sum(params.gmm_weights * params.gmm_vols**2)),
        rtol=0.0,
        atol=0.0,
    )
    params.sort_by_mus()
    assert np.all(np.diff(params.gmm_mus) >= 0.0)
    np.testing.assert_array_equal(params.gmm_weights, np.array([0.4, 0.6]))


def test_one_state_gmm_calibration_recovers_synthetic_surface() -> None:
    """The constrained GMM calibration recovers exact one-state Black quotes."""
    ttm = 0.5
    true_vol = 0.3
    true_params = GmmParams(
        gmm_weights=np.array([1.0]),
        gmm_mus=np.array([-0.5 * true_vol**2]),
        gmm_vols=np.array([true_vol]),
        ttm=ttm,
    )
    chain = _synthetic_gmm_slice(true_params)
    initial_vol = 0.27
    initial = GmmParams(
        gmm_weights=np.array([1.0]),
        gmm_mus=np.array([-0.5 * initial_vol**2]),
        gmm_vols=np.array([initial_vol]),
        ttm=ttm,
    )

    fitted = GmmPricer().calibrate_model_params_to_chain_slice(
        option_chain=chain,
        params0=initial,
        is_vega_weighted=False,
    )
    fitted_ivols = np.asarray(
        GmmPricer().compute_model_ivols_for_chain(chain, fitted)[0]
    )

    np.testing.assert_allclose(fitted.gmm_weights, 1.0, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(fitted.gmm_mus, -0.5 * fitted.gmm_vols**2, atol=1.0e-9)
    np.testing.assert_allclose(fitted.gmm_vols, true_vol, rtol=0.0, atol=2.0e-5)
    np.testing.assert_allclose(fitted_ivols, chain.get_mid_vols()[0], atol=1.5e-5)


def test_student_t_prices_satisfy_discounted_put_call_parity() -> None:
    """Student-t call and put prices preserve the forward martingale identity."""
    ttm = 0.5
    forward = 1.03
    strike = 1.0
    discfactor = 0.98
    rf_rate = -np.log(discfactor) / ttm
    params = TdistParams(
        drift=imply_drift_tdist(rf_rate=rf_rate, vol=0.3, nu=5.0, ttm=ttm),
        vol=0.3,
        nu=5.0,
        ttm=ttm,
    )
    pricer = TdistPricer()

    call, _ = pricer.price_vanilla(
        params=params,
        ttm=ttm,
        forward=forward,
        strike=strike,
        optiontype="C",
        discfactor=discfactor,
    )
    put, _ = pricer.price_vanilla(
        params=params,
        ttm=ttm,
        forward=forward,
        strike=strike,
        optiontype="P",
        discfactor=discfactor,
    )

    np.testing.assert_allclose(
        call - put,
        discfactor * (forward - strike),
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_student_t_distribution_density_cdf_and_forward_identities() -> None:
    vol = 0.3
    nu = 5.0
    ttm = 0.5
    rate = 0.02
    scale = tdist_fitter.compute_upsilon.py_func(vol, ttm, nu)
    np.testing.assert_allclose(
        scale**2 * nu / (nu - 2.0),
        vol**2 * ttm,
        rtol=0.0,
        atol=1.0e-15,
    )
    with pytest.raises(ValueError, match="> 2"):
        tdist_fitter.compute_upsilon.py_func(vol, ttm, 2.0)

    grid = np.linspace(-8.0, 8.0, 100_001)
    density = tdist_fitter.pdf_tdist(grid, mu=0.0, vol=vol, nu=nu, ttm=ttm)
    cdf = tdist_fitter.cdf_tdist(grid, mu=0.0, vol=vol, nu=nu, ttm=ttm)
    np.testing.assert_allclose(np.trapezoid(density, grid), 1.0, atol=2.0e-8)
    np.testing.assert_allclose(tdist_fitter.cdf_tdist(0.0, 0.0, vol, nu, ttm), 0.5)
    assert np.all(np.diff(cdf) >= 0.0)
    assert tdist_fitter.cum_mean_tdist(0.0, 0.0, vol, nu, ttm) < 0.0

    forward = tdist_fitter.compute_forward_tdist(
        spot=1.0,
        ttm=ttm,
        vol=vol,
        nu=nu,
        rf_rate=rate,
    )
    default_probability = tdist_fitter.compute_default_prob_tdist(
        ttm=ttm,
        vol=vol,
        nu=nu,
        rf_rate=rate,
    )
    np.testing.assert_allclose(forward, np.exp(rate * ttm), rtol=0.0, atol=2.0e-12)
    assert 0.0 < default_probability < 1.0e-3


def test_student_t_vector_pricing_and_implied_vol_round_trip() -> None:
    ttm = 0.5
    spot = 0.99
    rate = 0.02
    vol = 0.3
    nu = 5.0
    strikes = np.array([0.9, 1.0, 1.1])
    optiontypes = np.array(["P", "C", "C"])
    drift = tdist_fitter.imply_drift_tdist(rate, vol, nu, ttm)
    prices = tdist_fitter.compute_vanilla_price_tdist(
        spot=spot,
        strikes=strikes,
        ttm=ttm,
        vol=vol,
        nu=nu,
        optiontypes=optiontypes,
        rf_rate=rate,
        risk_neutral_mu=drift,
    )
    scalar_prices = np.array(
        [
            tdist_fitter.compute_vanilla_price_tdist(
                spot=spot,
                strikes=strike,
                ttm=ttm,
                vol=vol,
                nu=nu,
                optiontypes=optiontype,
                rf_rate=rate,
                is_compute_risk_neutral_mu=False,
                risk_neutral_mu=drift,
            )
            for strike, optiontype in zip(strikes, optiontypes)
        ]
    )
    recovered = tdist_fitter.infer_tdist_implied_vols_from_model_slice_prices(
        ttm=ttm,
        spot=spot,
        strikes=strikes,
        optiontypes=optiontypes,
        model_prices=prices,
        rf_rate=rate,
        nu=nu,
    )

    np.testing.assert_allclose(scalar_prices, prices, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(recovered, vol, rtol=0.0, atol=2.0e-10)
    low_bound = tdist_fitter.infer_implied_vol_tdist(
        spot=spot,
        ttm=ttm,
        strike=1.0,
        given_price=-1.0,
        rf_rate=rate,
        optiontype="C",
        nu=nu,
        is_bounds_to_nan=True,
    )
    assert np.isnan(low_bound)
    with pytest.raises(NotImplementedError, match="optiontype"):
        tdist_fitter.compute_vanilla_price_tdist(
            spot=spot,
            strikes=1.0,
            ttm=ttm,
            vol=vol,
            nu=nu,
            optiontypes="BAD",
            rf_rate=rate,
        )


def test_student_t_calibration_recovers_synthetic_surface() -> None:
    """A successful calibration must recover exact synthetic Student-t quotes."""
    ttm = 0.5
    true_params = TdistParams(
        drift=imply_drift_tdist(rf_rate=0.02, vol=0.3, nu=5.0, ttm=ttm),
        vol=0.3,
        nu=5.0,
        ttm=ttm,
    )
    chain = _synthetic_tdist_slice(true_params)
    initial = TdistParams(
        drift=imply_drift_tdist(rf_rate=0.02, vol=0.27, nu=6.0, ttm=ttm),
        vol=0.27,
        nu=6.0,
        ttm=ttm,
    )

    fitted = TdistPricer().calibrate_model_params_to_chain_slice(
        option_chain=chain,
        params0=initial,
        is_vega_weighted=False,
    )
    fitted_ivols = np.asarray(
        TdistPricer().compute_model_ivols_for_chain(chain, fitted)[0]
    )

    np.testing.assert_allclose(fitted.vol, true_params.vol, rtol=0.0, atol=2.0e-5)
    np.testing.assert_allclose(fitted.nu, true_params.nu, rtol=0.0, atol=2.0e-3)
    np.testing.assert_allclose(
        fitted_ivols,
        chain.get_mid_vols()[0],
        rtol=0.0,
        atol=2.0e-6,
    )


@pytest.mark.parametrize(
    ("pricer", "params"),
    [
        (
            GmmPricer(),
            GmmParams(
                gmm_weights=np.array([1.0]),
                gmm_mus=np.array([-0.5 * 0.3**2]),
                gmm_vols=np.array([0.3]),
                ttm=0.5,
            ),
        ),
        (
            TdistPricer(),
            TdistParams(
                drift=imply_drift_tdist(rf_rate=0.02, vol=0.3, nu=5.0, ttm=0.5),
                vol=0.3,
                nu=5.0,
                ttm=0.5,
            ),
        ),
    ],
)
def test_distribution_chain_slice_and_vanilla_interfaces_are_consistent(
    pricer,
    params,
) -> None:
    chain = OptionChain.slice_to_chain(
        ttm=0.5,
        forward=1.0,
        strikes=np.array([0.9, 1.0, 1.1]),
        optiontypes=np.array(["P", "C", "C"]),
        discfactor=np.exp(-0.01),
        id="6m",
    )
    chain_prices = np.asarray(pricer.price_chain(chain, params)[0])
    slice_prices, slice_ivols = pricer.price_slice(
        params=params,
        ttm=chain.ttms[0],
        forward=chain.forwards[0],
        strikes=np.asarray(chain.strikes_ttms[0]),
        optiontypes=np.asarray(chain.optiontypes_ttms[0]),
        discfactor=chain.discfactors[0],
    )

    np.testing.assert_allclose(slice_prices, chain_prices, rtol=0.0, atol=1.0e-14)
    for index, (strike, optiontype) in enumerate(
        zip(chain.strikes_ttms[0], chain.optiontypes_ttms[0])
    ):
        price, ivol = pricer.price_vanilla(
            params=params,
            ttm=chain.ttms[0],
            forward=chain.forwards[0],
            strike=strike,
            optiontype=optiontype,
            discfactor=chain.discfactors[0],
        )
        np.testing.assert_allclose(price, chain_prices[index], rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(ivol, slice_ivols[index], rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize("pricer", [GmmPricer(), TdistPricer()])
def test_distribution_slice_calibration_rejects_multiple_maturities(pricer) -> None:
    chain = OptionChain.get_uniform_chain(
        ttms=np.array([0.25, 0.5]),
        ids=np.array(["3m", "6m"]),
        forwards=np.ones(2),
        strikes=np.array([0.9, 1.0, 1.1]),
        flat_vol=0.2,
    )

    with pytest.raises(NotImplementedError, match="multiple slices"):
        pricer.calibrate_model_params_to_chain_slice(chain)


@pytest.mark.parametrize(
    ("module", "pricer", "kwargs", "size"),
    [
        (gmm_module, GmmPricer(), {"n_mixtures": 2}, 6),
        (tdist_module, TdistPricer(), {}, 2),
    ],
)
def test_distribution_calibrators_reject_failed_optimizer_results(
    monkeypatch: pytest.MonkeyPatch,
    module,
    pricer,
    kwargs: dict,
    size: int,
) -> None:
    """A failed optimizer must not be converted silently into public model parameters."""

    def failed_minimize(*args, **optimizer_kwargs):
        return SimpleNamespace(
            success=False,
            message="forced optimizer failure",
            x=np.zeros(size),
        )

    monkeypatch.setattr(module, "minimize", failed_minimize)

    with pytest.raises(CalibrationError, match="forced optimizer failure"):
        pricer.calibrate_model_params_to_chain_slice(
            option_chain=_quoted_slice(),
            is_vega_weighted=False,
            **kwargs,
        )
