"""Independent contracts for the stable Gaussian-mixture and Student-t pricers."""

from types import SimpleNamespace

import numpy as np
import pytest
from numba.typed import List
from scipy.integrate import quad
from scipy.stats import norm, t as student_t

import stochvolmodels
import stochvolmodels.pricers.gmm_pricer as gmm_module
import stochvolmodels.pricers.tdist_pricer as tdist_module
import stochvolmodels.fitters.tdist as tdist_fitter
from stochvolmodels import (
    CalibrationError,
    GmmParams,
    GmmPricer,
    OptionChain,
    OptionSlice,
    TdistParams,
    TdistPricer,
    compute_bsm_vanilla_price,
)
from stochvolmodels.fitters.tdist import imply_drift_tdist
from stochvolmodels.models import (
    PathModel,
    TerminalDistributionModel,
    TerminalSmileModel,
    TransformModel,
)
from stochvolmodels.pricers.gmm_pricer import (
    GmmTerminalModel,
    compute_gmm_vanilla_price,
    compute_gmm_vanilla_slice_prices,
    gmm_vanilla_chain_pricer,
)
from stochvolmodels.pricers.tdist_pricer import TdistTerminalModel


_TDIST_TTM = 0.5
_TDIST_FORWARD = 1.0
_TDIST_DISCOUNT_FACTOR = np.exp(-0.01)
_TDIST_DRIFT = 0.019643159690483324
_TDIST_STRIKES = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
_TDIST_OPTIONTYPES = np.array(["P", "P", "C", "C", "C"])
_TDIST_PRICES = np.array(
    [
        0.01778114786520990,
        0.03790196930809202,
        0.07631380572251986,
        0.03799929065161961,
        0.01794390178480159,
    ]
)
_TDIST_BLACK_IVOLS = np.array(
    [
        0.32463119501585250,
        0.29311367679985950,
        0.27367063936838343,
        0.26544693479584630,
        0.26590566227895550,
    ]
)

_GMM_TTM = 0.75
_GMM_FORWARD = 105.0
_GMM_DISCOUNT_FACTOR = 0.96
_GMM_WEIGHTS = np.array([0.4, 0.6])
_GMM_MUS = np.array([-0.18, -0.004343311874409908])
_GMM_VOLS = np.array([0.2, 0.45])
_GMM_STRIKES = np.array([80.0, 105.0, 130.0])
_GMM_OPTIONTYPES = np.array(["P", "C", "C"])
_GMM_PRICES = np.array(
    [2.779451417834116, 13.090788293511062, 6.651985427873406]
)
_GMM_BLACK_IVOLS = np.array(
    [0.3552383594248255, 0.3775686095394237, 0.4102391784434851]
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


def _tdist_terminal_model(**overrides: float) -> TdistTerminalModel:
    """Return the frozen-reference Student-t terminal model with optional overrides."""
    values = {
        "drift": _TDIST_DRIFT,
        "vol": 0.3,
        "nu": 5.0,
        "ttm": _TDIST_TTM,
    }
    values.update(overrides)
    return TdistTerminalModel(params=TdistParams(**values))


def _tdist_terminal_slice(
    *,
    strikes: np.ndarray = _TDIST_STRIKES,
    optiontypes: np.ndarray = _TDIST_OPTIONTYPES,
    ttm: float = _TDIST_TTM,
    discfactor: float = _TDIST_DISCOUNT_FACTOR,
) -> OptionSlice:
    """Return the frozen-reference option slice."""
    return OptionSlice(
        ttm=ttm,
        forward=_TDIST_FORWARD,
        strikes=strikes,
        optiontypes=optiontypes,
        id="student_t_reference",
        discfactor=discfactor,
    )


def _integrated_tdist_prices(
    model: TdistTerminalModel,
    option_slice: OptionSlice,
) -> np.ndarray:
    """Integrate payoffs directly under SciPy's Student-t density and the zero atom."""
    params = model.params
    spot = option_slice.forward * option_slice.discfactor
    scale = params.vol * np.sqrt(params.ttm * (params.nu - 2.0) / params.nu)
    law = student_t(df=params.nu, loc=0.0, scale=scale)
    default_boundary = -(1.0 + params.drift * params.ttm)
    default_probability = law.cdf(default_boundary)
    prices = np.empty(option_slice.strikes.size, dtype=float)

    for index, (strike, optiontype) in enumerate(
        zip(option_slice.strikes, option_slice.optiontypes)
    ):
        exercise_boundary = strike / spot - (1.0 + params.drift * params.ttm)
        if optiontype == "C":
            undiscounted, _ = quad(
                lambda shock: (
                    spot * (1.0 + params.drift * params.ttm + shock) - strike
                )
                * law.pdf(shock),
                exercise_boundary,
                np.inf,
                epsabs=1.0e-13,
                epsrel=1.0e-13,
                limit=200,
            )
        else:
            continuous, _ = quad(
                lambda shock: (
                    strike - spot * (1.0 + params.drift * params.ttm + shock)
                )
                * law.pdf(shock),
                default_boundary,
                exercise_boundary,
                epsabs=1.0e-13,
                epsrel=1.0e-13,
                limit=200,
            )
            undiscounted = strike * default_probability + continuous
        prices[index] = option_slice.discfactor * undiscounted

    assert default_probability > 0.0
    return prices


def _gmm_terminal_params(**overrides: object) -> GmmParams:
    """Return a detached copy of the frozen martingale Gaussian-mixture fixture."""
    values = {
        "gmm_weights": _GMM_WEIGHTS.copy(),
        "gmm_mus": _GMM_MUS.copy(),
        "gmm_vols": _GMM_VOLS.copy(),
        "ttm": _GMM_TTM,
    }
    values.update(overrides)
    return GmmParams(**values)


def _gmm_terminal_model(**overrides: object) -> GmmTerminalModel:
    """Return the validated terminal adapter for the frozen GMM fixture."""
    return GmmTerminalModel(params=_gmm_terminal_params(**overrides))


def _gmm_terminal_slice(
    *,
    strikes: np.ndarray = _GMM_STRIKES,
    optiontypes: np.ndarray = _GMM_OPTIONTYPES,
    ttm: float = _GMM_TTM,
    forward: float = _GMM_FORWARD,
    discfactor: float = _GMM_DISCOUNT_FACTOR,
) -> OptionSlice:
    """Return one option slice under the frozen GMM forward convention."""
    return OptionSlice(
        ttm=ttm,
        forward=forward,
        strikes=strikes,
        optiontypes=optiontypes,
        id="gmm_reference",
        discfactor=discfactor,
    )


def _integrated_gmm_prices(
    model: GmmTerminalModel,
    option_slice: OptionSlice,
) -> np.ndarray:
    """Integrate European payoffs directly against each Gaussian state density."""
    params = model.params
    prices = np.empty(option_slice.strikes.size, dtype=float)
    for index, (strike, optiontype) in enumerate(
        zip(option_slice.strikes, option_slice.optiontypes)
    ):
        exercise_boundary = np.log(strike / option_slice.forward)
        undiscounted = 0.0
        for weight, mu, vol in zip(
            params.gmm_weights,
            params.gmm_mus,
            params.gmm_vols,
        ):
            law = norm(loc=mu * params.ttm, scale=vol * np.sqrt(params.ttm))
            if optiontype == "C":
                upper = max(
                    exercise_boundary + 14.0 * law.std(),
                    law.mean() + law.var() + 14.0 * law.std(),
                )
                component, _ = quad(
                    lambda x: (option_slice.forward * np.exp(x) - strike) * law.pdf(x),
                    exercise_boundary,
                    upper,
                    epsabs=1.0e-12,
                    epsrel=1.0e-12,
                    limit=200,
                )
            else:
                lower = min(exercise_boundary - 14.0 * law.std(), law.mean() - 14.0 * law.std())
                component, _ = quad(
                    lambda x: (strike - option_slice.forward * np.exp(x)) * law.pdf(x),
                    lower,
                    exercise_boundary,
                    epsabs=1.0e-12,
                    epsrel=1.0e-12,
                    limit=200,
                )
            undiscounted += weight * component
        prices[index] = option_slice.discfactor * undiscounted
    return prices


def _integrated_gmm_mgf(model: GmmTerminalModel, phi: complex) -> complex:
    """Numerically integrate one real or complex exponential moment."""
    params = model.params
    value = 0.0j
    for weight, mu, vol in zip(
        params.gmm_weights,
        params.gmm_mus,
        params.gmm_vols,
    ):
        law = norm(loc=mu * params.ttm, scale=vol * np.sqrt(params.ttm))
        lower = law.mean() - 14.0 * law.std()
        upper = law.mean() + 14.0 * law.std()
        real_part, _ = quad(
            lambda x: np.real(np.exp(phi * x)) * law.pdf(x),
            lower,
            upper,
            epsabs=2.0e-13,
            epsrel=2.0e-13,
            limit=200,
        )
        imaginary_part, _ = quad(
            lambda x: np.imag(np.exp(phi * x)) * law.pdf(x),
            lower,
            upper,
            epsabs=2.0e-13,
            epsrel=2.0e-13,
            limit=200,
        )
        value += weight * complex(real_part, imaginary_part)
    return value


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


def test_gmm_terminal_model_matches_frozen_analytics_and_capabilities() -> None:
    """The adapter preserves the captured law and exposes only proven capabilities."""
    model = _gmm_terminal_model()
    option_slice = _gmm_terminal_slice()

    prices = model.price_european(option_slice)
    ivols = model.implied_vols(option_slice)

    assert isinstance(model, TerminalDistributionModel)
    assert isinstance(model, TerminalSmileModel)
    assert isinstance(model, TransformModel)
    assert not isinstance(model, PathModel)
    assert "GmmTerminalModel" not in stochvolmodels.__all__
    assert model.ttm == _GMM_TTM
    assert prices.shape == option_slice.strikes.shape
    assert ivols.shape == option_slice.strikes.shape
    assert np.issubdtype(prices.dtype, np.floating)
    assert np.issubdtype(ivols.dtype, np.floating)
    np.testing.assert_allclose(prices, _GMM_PRICES, rtol=0.0, atol=3.0e-13)
    np.testing.assert_allclose(ivols, _GMM_BLACK_IVOLS, rtol=0.0, atol=5.0e-12)


def test_gmm_terminal_prices_match_legacy_and_independent_density_quadrature() -> None:
    """The bound facade delegates unchanged analytics that direct payoff integration confirms."""
    model = _gmm_terminal_model()
    option_slice = _gmm_terminal_slice()
    chain = OptionChain.slice_to_chain(
        ttm=option_slice.ttm,
        forward=option_slice.forward,
        strikes=option_slice.strikes,
        optiontypes=option_slice.optiontypes,
        discfactor=option_slice.discfactor,
        id=option_slice.id,
    )
    legacy = np.asarray(GmmPricer().price_chain(chain, model.params)[0], dtype=float)

    np.testing.assert_allclose(model.price_european(option_slice), legacy, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        model.price_european(option_slice),
        _integrated_gmm_prices(model, option_slice),
        rtol=2.0e-12,
        atol=3.0e-12,
    )


def test_gmm_terminal_prices_obey_parity_monotonicity_and_convexity() -> None:
    """The martingale mixture produces an arbitrage-free standard-payoff strike slice."""
    model = _gmm_terminal_model()
    strikes = np.linspace(50.0, 160.0, 23)
    calls = model.price_european(
        _gmm_terminal_slice(strikes=strikes, optiontypes=np.array(["C"] * strikes.size))
    )
    puts = model.price_european(
        _gmm_terminal_slice(strikes=strikes, optiontypes=np.array(["P"] * strikes.size))
    )

    np.testing.assert_allclose(
        calls - puts,
        _GMM_DISCOUNT_FACTOR * (_GMM_FORWARD - strikes),
        rtol=0.0,
        atol=3.0e-13,
    )
    roundoff_tolerance = 3.0e-13
    assert np.max(np.diff(calls)) <= roundoff_tolerance
    assert np.min(np.diff(puts)) >= -roundoff_tolerance
    assert np.min(np.diff(calls, n=2)) >= -roundoff_tolerance
    assert np.min(np.diff(puts, n=2)) >= -roundoff_tolerance


def test_gmm_terminal_model_returns_float_arrays_for_integer_strikes() -> None:
    """The adapter must bypass the legacy integer ``zeros_like`` truncation."""
    model = _gmm_terminal_model()
    integer_slice = _gmm_terminal_slice(
        strikes=np.array([1, 2]),
        optiontypes=np.array(["C", "C"]),
        forward=1.0,
    )
    prices = model.price_european(integer_slice)
    ivols = model.implied_vols(integer_slice)

    assert np.issubdtype(prices.dtype, np.floating)
    assert np.issubdtype(ivols.dtype, np.floating)
    np.testing.assert_allclose(
        prices,
        np.array([0.12467417422391491, 0.00768647394089092]),
        rtol=0.0,
        atol=3.0e-14,
    )


def test_gmm_terminal_model_requires_exact_slice_maturity() -> None:
    mismatched = np.nextafter(_GMM_TTM, np.inf)
    with pytest.raises(ValueError, match="exactly match"):
        _gmm_terminal_model().price_european(_gmm_terminal_slice(ttm=mismatched))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gmm_weights", np.array([]), "gmm_weights"),
        ("gmm_weights", np.array([[0.4, 0.6]]), "gmm_weights"),
        ("gmm_weights", np.array([0.4, -0.6]), "nonnegative"),
        ("gmm_weights", np.array([0.3, 0.6]), "sum to one"),
        ("gmm_weights", np.array([True, False]), "gmm_weights"),
        ("gmm_weights", [0.4, True], "booleans"),
        ("gmm_weights", np.array([0.4, np.nan]), "gmm_weights"),
        ("gmm_mus", np.array([-0.18]), "same length"),
        ("gmm_mus", np.array([-0.18, np.inf]), "gmm_mus"),
        ("gmm_mus", np.array([-0.18 + 0.0j, 0.0j]), "gmm_mus"),
        ("gmm_vols", np.array([0.2]), "same length"),
        ("gmm_vols", np.array([0.2, 0.0]), "positive"),
        ("gmm_vols", np.array([0.2, np.nan]), "gmm_vols"),
        ("ttm", 0.0, "ttm"),
        ("ttm", -1.0, "ttm"),
        ("ttm", np.inf, "ttm"),
        ("ttm", True, "ttm"),
        ("gmm_mus", np.array([0.0, 0.0]), "martingale"),
    ],
)
def test_gmm_terminal_model_rejects_invalid_parameters(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _gmm_terminal_model(**{field: value})


def test_gmm_terminal_model_requires_gmm_params() -> None:
    with pytest.raises(TypeError, match="GmmParams"):
        GmmTerminalModel(params=object())


def test_gmm_terminal_model_detaches_caller_and_returned_parameters() -> None:
    """Neither legacy payload mutation route may alter the bound law."""
    params = _gmm_terminal_params()
    model = GmmTerminalModel(params=params)
    option_slice = _gmm_terminal_slice()
    prices = model.price_european(option_slice)

    params.gmm_weights[:] = [1.0, 0.0]
    params.gmm_mus[:] = 5.0
    params.gmm_vols[:] = 2.0
    params.ttm = 2.0
    inspected = model.params
    inspected.gmm_weights[:] = [1.0, 0.0]
    inspected.gmm_mus[:] = 5.0
    inspected.gmm_vols[:] = 2.0
    inspected.ttm = 2.0

    np.testing.assert_array_equal(model.params.gmm_weights, _GMM_WEIGHTS)
    np.testing.assert_array_equal(model.params.gmm_mus, _GMM_MUS)
    np.testing.assert_array_equal(model.params.gmm_vols, _GMM_VOLS)
    assert model.params.ttm == _GMM_TTM
    np.testing.assert_allclose(model.price_european(option_slice), prices, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("optiontype", ["IC", "IP"])
def test_gmm_terminal_model_does_not_claim_inverse_payoff_support(optiontype: str) -> None:
    with pytest.raises(NotImplementedError, match="inverse"):
        _gmm_terminal_model().price_european(
            _gmm_terminal_slice(
                strikes=np.array([105.0]),
                optiontypes=np.array([optiontype]),
            )
        )


def test_gmm_terminal_log_mgf_matches_real_and_complex_density_quadrature() -> None:
    """The closed-form transform agrees with integration independent of pricing kernels."""
    model = _gmm_terminal_model()
    real_grid = np.array([-0.5, 0.0, 0.4, 1.0, 2.0])
    complex_grid = np.array([-0.25 + 0.7j, 0.5 + 1.2j, 1.0 - 0.8j])

    real_values = model.log_mgf_grid(phi_grid=real_grid)
    complex_values = model.log_mgf_grid(phi_grid=complex_grid)
    integrated_real = np.array([np.log(_integrated_gmm_mgf(model, phi).real) for phi in real_grid])
    integrated_complex = np.array(
        [np.log(_integrated_gmm_mgf(model, phi)) for phi in complex_grid]
    )

    assert real_values.shape == real_grid.shape
    assert complex_values.shape == complex_grid.shape
    assert np.issubdtype(real_values.dtype, np.floating)
    assert np.issubdtype(complex_values.dtype, np.complexfloating)
    np.testing.assert_allclose(real_values, integrated_real, rtol=0.0, atol=3.0e-12)
    np.testing.assert_allclose(complex_values, integrated_complex, rtol=0.0, atol=3.0e-12)
    np.testing.assert_allclose(real_values[[1, 3]], 0.0, rtol=0.0, atol=5.0e-10)


def test_gmm_terminal_log_mgf_is_stable_for_large_transform_arguments() -> None:
    model = _gmm_terminal_model()
    phi_grid = np.array([100.0 + 0.0j, -100.0 + 0.0j, 100.0 + 1.0j])
    expected = np.array(
        [
            758.5384259856532 + 0.0j,
            759.1899227668148 + 0.0j,
            758.4624884856532 + 2.617871901735018j,
        ]
    )
    np.testing.assert_allclose(
        model.log_mgf_grid(phi_grid=phi_grid),
        expected,
        rtol=0.0,
        atol=2.0e-13,
    )


def test_gmm_terminal_log_mgf_ignores_zero_weight_states_when_stabilizing() -> None:
    """A numerically dominant zero-probability state cannot erase the active transform."""
    ttm = 0.75
    vol = 0.2
    model = GmmTerminalModel(
        GmmParams(
            gmm_weights=np.array([1.0, 0.0]),
            gmm_mus=np.array([-0.5 * vol**2, 1_000.0]),
            gmm_vols=np.array([vol, 1_000.0]),
            ttm=ttm,
        )
    )
    phi_grid = np.array([100.0, -100.0])
    expected = ttm * (-0.5 * vol**2 * phi_grid + 0.5 * vol**2 * phi_grid**2)
    np.testing.assert_allclose(
        model.log_mgf_grid(phi_grid=phi_grid),
        expected,
        rtol=0.0,
        atol=2.0e-13,
    )


@pytest.mark.parametrize(
    "phi_grid",
    [
        np.array([]),
        np.array([True, False]),
        np.array(["0", "1"]),
        np.array([np.nan]),
        np.array([1.0 + np.nan * 1.0j]),
    ],
)
def test_gmm_terminal_log_mgf_rejects_invalid_grids(phi_grid: np.ndarray) -> None:
    with pytest.raises(ValueError, match="phi_grid"):
        _gmm_terminal_model().log_mgf_grid(phi_grid=phi_grid)


def test_gmm_terminal_transform_reproduces_paper_risk_premium_algebra() -> None:
    model = _gmm_terminal_model()
    kappa = 3.0
    log_moments = model.log_mgf_grid(phi_grid=np.array([kappa, kappa + 1.0]))
    risk_premium = (np.exp(log_moments[1] - log_moments[0]) - 1.0) / model.ttm
    np.testing.assert_allclose(risk_premium, 0.7287604830098268, rtol=0.0, atol=5.0e-14)


def test_gmm_calibration_rejects_constraint_violating_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optimizer success cannot override the probability and martingale constraints."""
    invalid = np.array([0.5, 0.5, 0.0, 0.0, 0.2, 0.3])

    def false_success(*args, **kwargs):
        return SimpleNamespace(success=True, message="false success", x=invalid, fun=0.0)

    monkeypatch.setattr(gmm_module, "minimize", false_success)
    with pytest.raises(CalibrationError, match="martingale"):
        GmmPricer().calibrate_model_params_to_chain_slice(
            option_chain=_quoted_slice(),
            n_mixtures=2,
            is_vega_weighted=False,
        )


def test_gmm_scalar_calibration_rejects_nonfinite_reported_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def false_success(*args, **kwargs):
        return SimpleNamespace(success=True, message="false success", x=0.3, fun=np.nan)

    monkeypatch.setattr(gmm_module, "minimize_scalar", false_success)
    with pytest.raises(CalibrationError, match="objective"):
        GmmPricer().calibrate_model_params_to_chain_slice(
            option_chain=_quoted_slice(),
            n_mixtures=1,
            is_vega_weighted=False,
        )


def test_gmm_scalar_calibration_rejects_all_nan_repricing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pricer = GmmPricer()

    def false_success(*args, **kwargs):
        return SimpleNamespace(success=True, message="false success", x=0.3, fun=0.0)

    def all_nan_ivols(*args, **kwargs):
        return (np.full(3, np.nan),)

    monkeypatch.setattr(gmm_module, "minimize_scalar", false_success)
    monkeypatch.setattr(pricer, "compute_model_ivols_for_chain", all_nan_ivols)
    with pytest.raises(CalibrationError, match="objective"):
        pricer.calibrate_model_params_to_chain_slice(
            option_chain=_quoted_slice(),
            n_mixtures=1,
            is_vega_weighted=False,
        )


def test_gmm_calibration_uses_only_finite_positive_weight_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inactive quote NaNs do not contaminate the finite calibration objective."""
    chain = _quoted_slice()
    pricer = GmmPricer()

    monkeypatch.setattr(
        chain,
        "get_chain_data_as_xy",
        lambda: (None, (np.array([np.nan, 0.23, 0.24]),)),
    )
    monkeypatch.setattr(
        pricer,
        "compute_model_ivols_for_chain",
        lambda *args, **kwargs: (np.array([np.nan, 0.23, 0.24]),),
    )
    monkeypatch.setattr(
        gmm_module,
        "minimize_scalar",
        lambda *args, **kwargs: SimpleNamespace(
            success=True,
            message="active quotes",
            x=0.3,
            fun=0.0,
        ),
    )

    fitted = pricer.calibrate_model_params_to_chain_slice(
        option_chain=chain,
        n_mixtures=1,
        is_vega_weighted=False,
    )
    np.testing.assert_allclose(fitted.gmm_vols, 0.3, rtol=0.0, atol=0.0)


def test_gmm_calibration_rejects_no_active_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    chain = _quoted_slice()
    monkeypatch.setattr(
        chain,
        "get_chain_data_as_xy",
        lambda: (None, (np.full(3, np.nan),)),
    )
    with pytest.raises(CalibrationError, match="no finite positive-weight quotes"):
        GmmPricer().calibrate_model_params_to_chain_slice(
            option_chain=chain,
            n_mixtures=1,
            is_vega_weighted=False,
        )


def test_gmm_calibration_rejects_invalid_vega_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _quoted_slice()
    monkeypatch.setattr(chain, "get_chain_vegas", lambda **kwargs: (np.zeros(3),))
    with pytest.raises(CalibrationError, match="invalid vega weights"):
        GmmPricer().calibrate_model_params_to_chain_slice(
            option_chain=chain,
            n_mixtures=1,
            is_vega_weighted=True,
        )


@pytest.mark.slow
def test_two_state_gmm_calibration_reprices_synthetic_holdout_strikes() -> None:
    """A valid multi-state fit retains the generating law away from calibration strikes."""
    true_model = _gmm_terminal_model()
    training_strikes = np.array([70.0, 85.0, 100.0, 115.0, 130.0, 145.0])
    training_types = np.where(training_strikes < _GMM_FORWARD, "P", "C")
    training_slice = _gmm_terminal_slice(
        strikes=training_strikes,
        optiontypes=training_types,
    )
    training_ivols = true_model.implied_vols(training_slice)
    quoted_chain = OptionChain(
        ttms=np.array([training_slice.ttm]),
        forwards=np.array([training_slice.forward]),
        strikes_ttms=List([training_strikes]),
        optiontypes_ttms=List([training_types]),
        ids=np.array([training_slice.id]),
        discfactors=np.array([training_slice.discfactor]),
        bid_ivs=List([training_ivols]),
        ask_ivs=List([training_ivols]),
    )

    fitted = GmmPricer().calibrate_model_params_to_chain_slice(
        option_chain=quoted_chain,
        params0=true_model.params,
        is_vega_weighted=False,
    )
    holdout_strikes = np.array([77.5, 92.5, 107.5, 122.5, 137.5])
    holdout_types = np.where(holdout_strikes < _GMM_FORWARD, "P", "C")
    holdout_slice = _gmm_terminal_slice(
        strikes=holdout_strikes,
        optiontypes=holdout_types,
    )

    np.testing.assert_allclose(
        GmmTerminalModel(fitted).implied_vols(holdout_slice),
        true_model.implied_vols(holdout_slice),
        rtol=0.0,
        atol=3.0e-7,
    )


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


def test_tdist_terminal_model_matches_frozen_prices_and_black_ivols() -> None:
    """The adapter preserves captured analytics and exposes both terminal capabilities."""
    model = _tdist_terminal_model()
    option_slice = _tdist_terminal_slice()

    prices = model.price_european(option_slice)
    ivols = model.implied_vols(option_slice)

    assert isinstance(model, TerminalDistributionModel)
    assert isinstance(model, TerminalSmileModel)
    assert not isinstance(model, PathModel)
    assert not isinstance(model, TransformModel)
    assert model.ttm == _TDIST_TTM
    assert prices.shape == option_slice.strikes.shape
    assert ivols.shape == option_slice.strikes.shape
    assert np.issubdtype(prices.dtype, np.floating)
    assert np.issubdtype(ivols.dtype, np.floating)
    np.testing.assert_allclose(prices, _TDIST_PRICES, rtol=0.0, atol=3.0e-14)
    np.testing.assert_allclose(ivols, _TDIST_BLACK_IVOLS, rtol=0.0, atol=5.0e-12)


def test_tdist_terminal_prices_match_independent_payoff_integration() -> None:
    """Closed-form prices agree with direct integration including the default atom."""
    model = _tdist_terminal_model()
    option_slice = _tdist_terminal_slice()

    np.testing.assert_allclose(
        model.price_european(option_slice),
        _integrated_tdist_prices(model, option_slice),
        rtol=2.0e-12,
        atol=2.0e-13,
    )


def test_tdist_terminal_prices_satisfy_discounted_put_call_parity() -> None:
    """Paired calls and puts preserve the adapter's forward convention."""
    model = _tdist_terminal_model()
    strikes = np.array([0.8, 1.0, 1.2])
    calls = model.price_european(
        _tdist_terminal_slice(strikes=strikes, optiontypes=np.array(["C"] * 3))
    )
    puts = model.price_european(
        _tdist_terminal_slice(strikes=strikes, optiontypes=np.array(["P"] * 3))
    )

    np.testing.assert_allclose(
        calls - puts,
        _TDIST_DISCOUNT_FACTOR * (_TDIST_FORWARD - strikes),
        rtol=2.0e-12,
        atol=2.0e-13,
    )


def test_tdist_terminal_prices_are_monotone_and_convex_in_strike() -> None:
    """Standard call and put grids satisfy their no-arbitrage strike-shape constraints."""
    model = _tdist_terminal_model()
    strikes = np.linspace(0.5, 1.5, 21)
    calls = model.price_european(
        _tdist_terminal_slice(strikes=strikes, optiontypes=np.array(["C"] * strikes.size))
    )
    puts = model.price_european(
        _tdist_terminal_slice(strikes=strikes, optiontypes=np.array(["P"] * strikes.size))
    )

    # This is about 900 float64 eps at unit scale and matches the direct-integration allowance.
    roundoff_tolerance = 2.0e-13
    assert np.max(np.diff(calls)) <= roundoff_tolerance
    assert np.min(np.diff(puts)) >= -roundoff_tolerance
    assert np.min(np.diff(calls, n=2)) >= -roundoff_tolerance
    assert np.min(np.diff(puts, n=2)) >= -roundoff_tolerance


def test_tdist_terminal_model_returns_float_arrays_for_integer_strikes() -> None:
    """Integer input storage must not truncate terminal prices or implied volatilities."""
    model = _tdist_terminal_model()
    integer_slice = _tdist_terminal_slice(
        strikes=np.array([1, 2]),
        optiontypes=np.array(["C", "C"]),
    )
    float_slice = _tdist_terminal_slice(
        strikes=np.array([1.0, 2.0]),
        optiontypes=np.array(["C", "C"]),
    )

    integer_prices = model.price_european(integer_slice)
    integer_ivols = model.implied_vols(integer_slice)

    assert np.issubdtype(integer_prices.dtype, np.floating)
    assert np.issubdtype(integer_ivols.dtype, np.floating)
    assert integer_prices[1] > 0.0
    np.testing.assert_allclose(
        integer_prices,
        model.price_european(float_slice),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        integer_ivols,
        model.implied_vols(float_slice),
        rtol=0.0,
        atol=0.0,
    )


def test_tdist_terminal_model_requires_exact_slice_maturity() -> None:
    """A one-maturity parameter set cannot silently price another maturity."""
    mismatched_ttm = np.nextafter(_TDIST_TTM, np.inf)

    with pytest.raises(ValueError, match="exactly match"):
        _tdist_terminal_model().price_european(_tdist_terminal_slice(ttm=mismatched_ttm))


def test_tdist_terminal_model_requires_consistent_discount_rate() -> None:
    """The slice discount rate must agree with the drift's floored martingale equation."""
    inconsistent_slice = _tdist_terminal_slice(discfactor=np.exp(-0.015))

    with pytest.raises(ValueError, match="discount rate"):
        _tdist_terminal_model().price_european(inconsistent_slice)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("drift", np.nan),
        ("drift", np.inf),
        ("drift", True),
        ("vol", 0.0),
        ("vol", -0.1),
        ("vol", np.nan),
        ("vol", np.bool_(True)),
        ("nu", 2.0),
        ("nu", 1.9),
        ("nu", np.inf),
        ("ttm", 0.0),
        ("ttm", -0.1),
        ("ttm", np.nan),
    ],
)
def test_tdist_terminal_model_rejects_invalid_parameters(field: str, value: float) -> None:
    """The adapter rejects non-finite or mathematically invalid bound parameters."""
    with pytest.raises(ValueError, match=field):
        _tdist_terminal_model(**{field: value})


def test_tdist_terminal_model_requires_tdist_params() -> None:
    """The bound parameter payload must use the established dataclass."""
    with pytest.raises(TypeError, match="TdistParams"):
        TdistTerminalModel(params=object())


def test_tdist_terminal_model_detaches_from_caller_params() -> None:
    """Mutating the caller's legacy payload cannot change the bound terminal law."""
    params = TdistParams(drift=_TDIST_DRIFT, vol=0.3, nu=5.0, ttm=_TDIST_TTM)
    model = TdistTerminalModel(params=params)
    option_slice = _tdist_terminal_slice()
    prices = model.price_european(option_slice)

    assert model.params is not params
    params.drift = 1.0
    params.vol = 1.0
    params.nu = 3.0
    params.ttm = 1.0

    assert model.params == TdistParams(
        drift=_TDIST_DRIFT,
        vol=0.3,
        nu=5.0,
        ttm=_TDIST_TTM,
    )
    np.testing.assert_allclose(model.price_european(option_slice), prices, rtol=0.0, atol=0.0)


def test_tdist_terminal_model_detaches_returned_params() -> None:
    """Mutating an inspected parameter copy cannot change the bound terminal law."""
    model = _tdist_terminal_model()
    option_slice = _tdist_terminal_slice()
    prices = model.price_european(option_slice)
    inspected_params = model.params

    inspected_params.drift = 1.0
    inspected_params.vol = 1.0
    inspected_params.nu = 3.0
    inspected_params.ttm = 1.0

    assert model.params == TdistParams(
        drift=_TDIST_DRIFT,
        vol=0.3,
        nu=5.0,
        ttm=_TDIST_TTM,
    )
    np.testing.assert_allclose(model.price_european(option_slice), prices, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("optiontype", ["IC", "IP"])
def test_tdist_terminal_model_does_not_claim_inverse_payoff_support(optiontype: str) -> None:
    """Historical inverse codes remain outside the new terminal-model contract."""
    option_slice = _tdist_terminal_slice(
        strikes=np.array([1.0]),
        optiontypes=np.array([optiontype]),
    )

    with pytest.raises(NotImplementedError, match="inverse"):
        _tdist_terminal_model().price_european(option_slice)


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
