"""Independent characterization of the inverse-gamma/normal terminal law."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
import vanilla_option_pricers as bsm
from scipy.integrate import quad
from scipy.special import gammaln, kve, ndtr

from stochvolmodels import OptionSlice
from stochvolmodels.fitters.tdist import cdf_tdist, imply_drift_tdist
from stochvolmodels.models import (
    PathModel,
    TerminalDistributionModel,
    TerminalSmileModel,
    TransformModel,
)
from stochvolmodels.models.inverse_gamma_normal import (
    InverseGammaNormalParams,
    InverseGammaNormalTerminalModel,
)
from stochvolmodels.pricers.tdist_pricer import TdistParams, TdistTerminalModel

_FROZEN_CURVE_HASH = "2f78c1e678774f692946fc37f97496c754a0f306604af136d5800a342fbefeae"
_CHAPTER_LAWS = (
    (4.75, 0.120, 0.0),
    (4.00, 0.120, 0.0),
    (3.25, 0.120, 0.0),
    (4.00, 0.096, 0.0),
    (4.00, 0.120, 0.0),
    (4.00, 0.144, 0.0),
    (4.00, 0.120, -2.0),
    (4.00, 0.120, 0.0),
    (4.00, 0.120, 2.0),
    (4.75, 0.150, 0.0),
    (4.00, 0.120, 0.0),
    (3.25, 0.090, 0.0),
)
_EXPECTED_SHIFTS = np.array(
    [
        0.9999920955561977,
        0.9999646285152991,
        0.9998319143510996,
        0.9999838027697864,
        0.9999646285152991,
        0.9999341249741032,
        1.079741087577782,
        0.9999646285152991,
        0.9199967270263122,
        0.9999807026924815,
        0.9999646285152991,
        0.9999266976092555,
    ]
)
_EXPECTED_DEFAULT_PROBABILITIES = np.array(
    [
        5.635434956124023e-05,
        2.0885213649389573e-04,
        7.858410761328880e-04,
        9.864535092085288e-05,
        2.0885213649389573e-04,
        3.7759424368605086e-04,
        9.935804567066705e-04,
        2.0885213649389573e-04,
        3.055402025491766e-05,
        1.324334797811496e-04,
        2.0885213649389573e-04,
        3.5571943435259675e-04,
    ]
)


def _model(
    *,
    alpha: float = 4.0,
    beta: float = 0.12,
    c: float = 1.0,
    q: float = 0.0,
    ttm: float = 1.0,
    quadrature_order: int = 256,
) -> InverseGammaNormalTerminalModel:
    return InverseGammaNormalTerminalModel(
        InverseGammaNormalParams(alpha=alpha, beta=beta, c=c, q=q, ttm=ttm),
        quadrature_order=quadrature_order,
    )


def _slice(
    *,
    ttm: float = 1.0,
    forward: float = 1.0,
    discfactor: float = 1.0,
    strikes: np.ndarray | None = None,
    optiontypes: np.ndarray | None = None,
) -> OptionSlice:
    if strikes is None:
        strikes = np.array([0.8, 1.0, 1.2])
    if optiontypes is None:
        optiontypes = np.array(["P", "C", "C"])
    return OptionSlice(
        ttm=ttm,
        forward=forward,
        strikes=strikes,
        optiontypes=optiontypes,
        discfactor=discfactor,
        id="inverse-gamma-normal",
    )


def _canonical_curve_payload() -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    log_moneyness = np.linspace(-0.35, 0.35, 29)
    strikes = np.exp(log_moneyness)
    optiontypes = np.array(["C"] * strikes.size)
    records = []
    shifts = []
    default_probabilities = []
    for alpha, beta, q in _CHAPTER_LAWS:
        model = _model(alpha=alpha, beta=beta, q=q)
        option_slice = _slice(strikes=strikes, optiontypes=optiontypes)
        calls = model.price_european(option_slice)
        ivols = model.implied_vols(option_slice)
        shift = model.martingale_shift(discfactor=1.0)
        default_probability = model.default_probability(discfactor=1.0)
        shifts.append(shift)
        default_probabilities.append(default_probability)
        records.append(
            {
                "alpha": round(alpha, 12),
                "beta": round(beta, 12),
                "q": round(q, 12),
                "shift": round(shift, 12),
                "default_probability": round(default_probability, 12),
                "call_prices": np.round(calls, 10).tolist(),
                "implied_volatilities": np.round(ivols, 10).tolist(),
            }
        )
    payload = {
        "log_moneyness": np.round(log_moneyness, 8).tolist(),
        "records": records,
    }
    return payload, np.asarray(shifts), np.asarray(default_probabilities)


def _normal_positive_part(mean: float, stdev: float) -> float:
    d = mean / stdev
    return float(mean * ndtr(d) + stdev * np.exp(-0.5 * d * d) / math.sqrt(2.0 * math.pi))


def _adaptive_normalized_call(
    model: InverseGammaNormalTerminalModel,
    *,
    normalized_strike: float,
    discfactor: float,
) -> float:
    params = model.params
    shift = model.martingale_shift(discfactor=discfactor)

    def integrand(precision: float) -> float:
        if precision <= 0.0:
            return 0.0
        variance = params.beta / precision
        positive_part = _normal_positive_part(
            shift - normalized_strike + params.q * variance,
            math.sqrt(params.c * variance),
        )
        if positive_part == 0.0:
            return 0.0
        log_density = (params.alpha - 1.0) * math.log(precision) - precision - gammaln(params.alpha)
        return math.exp(log_density) * positive_part

    return float(
        quad(integrand, 0.0, 1.0, epsabs=2.0e-11, epsrel=2.0e-11, limit=300)[0]
        + quad(
            integrand,
            1.0,
            np.inf,
            epsabs=2.0e-11,
            epsrel=2.0e-11,
            limit=300,
        )[0]
    )


def _skew_marginal_density(
    params: InverseGammaNormalParams,
    shock: float,
) -> float:
    radius_squared = shock * shock + 2.0 * params.beta * params.c
    radius = math.sqrt(radius_squared)
    absolute_q = abs(params.q)
    bessel_argument = absolute_q * radius / params.c
    if params.q * shock >= 0.0:
        exponential_term = -2.0 * params.beta * absolute_q / (radius + abs(shock))
    else:
        exponential_term = -absolute_q * (radius + abs(shock)) / params.c
    log_density = (
        math.log(2.0)
        + params.alpha * math.log(params.beta)
        - gammaln(params.alpha)
        - 0.5 * math.log(2.0 * math.pi * params.c)
        + exponential_term
        + 0.5 * (params.alpha + 0.5) * (math.log(params.q * params.q) - math.log(radius_squared))
        + math.log(kve(params.alpha + 0.5, bessel_argument))
    )
    return math.exp(log_density)


def _bessel_normalized_call(
    model: InverseGammaNormalTerminalModel,
    *,
    normalized_strike: float,
    discfactor: float,
) -> float:
    params = model.params
    shift = model.martingale_shift(discfactor=discfactor)
    threshold = normalized_strike - shift

    def integrand(shock: float) -> float:
        return max(shift + shock - normalized_strike, 0.0) * _skew_marginal_density(
            params,
            shock,
        )

    value = 0.0
    if threshold < 0.0:
        value += quad(
            integrand,
            threshold,
            0.0,
            epsabs=2.0e-11,
            epsrel=2.0e-11,
            limit=300,
        )[0]
        threshold = 0.0
    value += quad(
        integrand,
        threshold,
        np.inf,
        epsabs=2.0e-11,
        epsrel=2.0e-11,
        limit=300,
    )[0]
    return float(value)


def test_inverse_gamma_normal_matches_all_frozen_chapter_curves() -> None:
    """All 12 captured laws reproduce the frozen 29-point calls and Black smiles."""

    payload, shifts, default_probabilities = _canonical_curve_payload()
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()

    assert hashlib.sha256(encoded).hexdigest() == _FROZEN_CURVE_HASH
    np.testing.assert_allclose(shifts, _EXPECTED_SHIFTS, rtol=0.0, atol=5.0e-14)
    np.testing.assert_allclose(
        default_probabilities,
        _EXPECTED_DEFAULT_PROBABILITIES,
        rtol=0.0,
        atol=5.0e-14,
    )


def test_inverse_gamma_normal_exposes_only_terminal_capabilities() -> None:
    """The one-maturity law is a distribution/smile model, not a path or transform model."""

    model = _model()

    assert isinstance(model, TerminalDistributionModel)
    assert isinstance(model, TerminalSmileModel)
    assert not isinstance(model, PathModel)
    assert not isinstance(model, TransformModel)
    assert model.params == InverseGammaNormalParams(4.0, 0.12, 1.0, 0.0, 1.0)
    assert model.ttm == 1.0
    assert model.quadrature_order == 256
    assert model.mean_mixing_variance == 0.04
    assert not hasattr(model, "log_mgf_grid")


def test_q_zero_matches_tdist_at_nonunit_discount() -> None:
    """The symmetric law matches the accepted T2 adapter under prepaid-forward scaling."""

    alpha = 4.0
    beta = 0.12
    c = 1.0
    ttm = 0.75
    discfactor = 0.96
    model = _model(alpha=alpha, beta=beta, c=c, q=0.0, ttm=ttm)
    nu = 2.0 * alpha
    vol = math.sqrt(c * beta / ((alpha - 1.0) * ttm))
    rate = -math.log(discfactor) / ttm
    drift = imply_drift_tdist(rf_rate=rate, vol=vol, nu=nu, ttm=ttm)
    reference = TdistTerminalModel(TdistParams(drift=drift, vol=vol, nu=nu, ttm=ttm))
    option_slice = _slice(
        ttm=ttm,
        forward=105.0,
        discfactor=discfactor,
        strikes=np.array([80.0, 95.0, 105.0, 115.0, 130.0]),
        optiontypes=np.array(["P", "P", "C", "C", "C"]),
    )

    expected_shift = 1.0 + drift * ttm
    expected_default = cdf_tdist(
        x=-expected_shift,
        mu=0.0,
        vol=vol,
        nu=nu,
        ttm=ttm,
    )
    np.testing.assert_allclose(
        model.martingale_shift(discfactor=discfactor),
        expected_shift,
        rtol=0.0,
        atol=4.0e-8,
    )
    np.testing.assert_allclose(
        model.default_probability(discfactor=discfactor),
        expected_default,
        rtol=0.0,
        atol=4.0e-8,
    )
    np.testing.assert_allclose(
        model.price_european(option_slice),
        reference.price_european(option_slice),
        rtol=0.0,
        atol=4.0e-6,
    )
    np.testing.assert_allclose(
        model.implied_vols(option_slice),
        reference.implied_vols(option_slice),
        rtol=0.0,
        atol=2.0e-7,
    )


@pytest.mark.parametrize("q", [-2.0, 2.0])
def test_skew_calls_match_adaptive_precision_integration(q: float) -> None:
    """Production Gauss-Laguerre calls match an adaptive Gamma-precision integral."""

    model = _model(q=q)
    discfactor = 0.97
    forward = 1.2
    prepaid_forward = discfactor * forward
    normalized_strikes = np.array([0.8, 1.0, 1.2])
    strikes = prepaid_forward * normalized_strikes
    option_slice = _slice(
        forward=forward,
        discfactor=discfactor,
        strikes=strikes,
        optiontypes=np.array(["C", "C", "C"]),
    )
    expected = np.array(
        [
            discfactor
            * prepaid_forward
            * _adaptive_normalized_call(
                model,
                normalized_strike=float(strike),
                discfactor=discfactor,
            )
            for strike in normalized_strikes
        ]
    )

    np.testing.assert_allclose(
        model.price_european(option_slice),
        expected,
        rtol=0.0,
        atol=4.0e-8,
    )
    np.testing.assert_allclose(
        _adaptive_normalized_call(model, normalized_strike=0.0, discfactor=discfactor),
        1.0 / discfactor,
        rtol=0.0,
        atol=4.0e-8,
    )


@pytest.mark.parametrize("q", [-2.0, 2.0])
def test_skew_atm_call_matches_bessel_density_integration(q: float) -> None:
    """A direct Bessel-K marginal-density integral independently prices the skew law."""

    model = _model(q=q)
    discfactor = 0.97
    forward = 1.2
    prepaid_forward = discfactor * forward
    option_slice = _slice(
        forward=forward,
        discfactor=discfactor,
        strikes=np.array([prepaid_forward]),
        optiontypes=np.array(["C"]),
    )
    expected = (
        discfactor
        * prepaid_forward
        * _bessel_normalized_call(
            model,
            normalized_strike=1.0,
            discfactor=discfactor,
        )
    )

    np.testing.assert_allclose(
        model.price_european(option_slice)[0],
        expected,
        rtol=0.0,
        atol=4.0e-8,
    )


def test_prices_obey_parity_shape_refinement_and_black_round_trip() -> None:
    """The skew law is arbitrage-consistent and numerically converged on a strike grid."""

    strikes = np.linspace(0.55, 1.45, 25)
    calls_slice = _slice(strikes=strikes, optiontypes=np.array(["C"] * strikes.size))
    puts_slice = _slice(strikes=strikes, optiontypes=np.array(["P"] * strikes.size))
    model = _model(q=-2.0)
    calls = model.price_european(calls_slice)
    puts = model.price_european(puts_slice)

    np.testing.assert_allclose(calls - puts, 1.0 - strikes, rtol=0.0, atol=7.0e-14)
    assert np.max(np.diff(calls)) <= 2.0e-13
    assert np.min(np.diff(puts)) >= -2.0e-13
    assert np.min(np.diff(calls, n=2)) >= -2.0e-13
    assert np.min(np.diff(puts, n=2)) >= -2.0e-13

    lower_order = _model(q=-2.0, quadrature_order=128)
    representative = _slice(
        strikes=np.exp(np.array([-0.25, 0.0, 0.25])),
        optiontypes=np.array(["C", "C", "C"]),
    )
    np.testing.assert_allclose(
        model.price_european(representative),
        lower_order.price_european(representative),
        rtol=0.0,
        atol=3.0e-7,
    )

    ivols = model.implied_vols(calls_slice)
    repriced = np.array(
        [
            bsm.compute_bsm_vanilla_price(
                forward=1.0,
                strike=float(strike),
                ttm=1.0,
                vol=float(vol),
                optiontype="C",
                discfactor=1.0,
            )
            for strike, vol in zip(strikes, ivols)
        ]
    )
    np.testing.assert_allclose(repriced, calls, rtol=0.0, atol=5.0e-13)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alpha", 1.0),
        ("alpha", 0.9),
        ("alpha", np.nan),
        ("alpha", True),
        ("beta", 0.0),
        ("beta", -0.1),
        ("beta", np.inf),
        ("c", 0.0),
        ("c", np.nan),
        ("q", np.inf),
        ("q", np.bool_(True)),
        ("ttm", 0.0),
        ("ttm", -0.1),
        ("ttm", np.nan),
    ],
)
def test_model_rejects_invalid_parameters(field: str, value: float) -> None:
    """The package boundary rejects non-finite and mathematically invalid law inputs."""

    values = {"alpha": 4.0, "beta": 0.12, "c": 1.0, "q": 0.0, "ttm": 1.0}
    values[field] = value
    with pytest.raises(ValueError, match=field):
        InverseGammaNormalTerminalModel(InverseGammaNormalParams(**values))


@pytest.mark.parametrize("quadrature_order", [True, 0, 1, 2.5, np.nan])
def test_model_rejects_invalid_quadrature_order(quadrature_order: object) -> None:
    """The numerical integration rule requires a usable integer node count."""

    with pytest.raises(ValueError, match="quadrature_order"):
        _model(quadrature_order=quadrature_order)


def test_model_rejects_wrong_payload_maturity_payoff_and_discount() -> None:
    """One-maturity and standard-payoff boundaries fail closed."""

    with pytest.raises(TypeError, match="InverseGammaNormalParams"):
        InverseGammaNormalTerminalModel(object())
    with pytest.raises(TypeError, match="OptionSlice"):
        _model().price_european(object())
    with pytest.raises(ValueError, match="exactly match"):
        _model().price_european(_slice(ttm=np.nextafter(1.0, np.inf)))
    with pytest.raises(ValueError, match="finite positive"):
        _model().price_european(_slice(ttm=True))
    with pytest.raises(NotImplementedError, match="inverse"):
        _model().price_european(_slice(strikes=np.array([1.0]), optiontypes=np.array(["IC"])))
    with pytest.raises(ValueError, match="discfactor"):
        _model().martingale_shift(discfactor=True)


def test_integer_strikes_return_floating_prices_and_ivols() -> None:
    """Integer strike storage cannot truncate prices or Black implied volatilities."""

    model = _model(q=2.0)
    integer_slice = _slice(
        forward=100.0,
        strikes=np.array([80, 100, 120]),
        optiontypes=np.array(["P", "C", "C"]),
    )
    float_slice = _slice(
        forward=100.0,
        strikes=np.array([80.0, 100.0, 120.0]),
        optiontypes=np.array(["P", "C", "C"]),
    )
    integer_prices = model.price_european(integer_slice)
    integer_ivols = model.implied_vols(integer_slice)

    assert np.issubdtype(integer_prices.dtype, np.floating)
    assert np.issubdtype(integer_ivols.dtype, np.floating)
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
