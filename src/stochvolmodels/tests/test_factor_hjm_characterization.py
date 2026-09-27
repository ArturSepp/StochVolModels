import numpy as np
import pytest

import stochvolmodels as svm
from stochvolmodels.pricers.factor_hjm.rate_affine_expansion import (
    UnderlyingType,
    compute_logsv_a_mgf_grid,
    func_a_ode_quadratic_terms,
)
from stochvolmodels.pricers.logsv.affine_expansion import ExpansionOrder


def _mgf_inputs() -> dict:
    times = np.linspace(0.0, 0.25, 5)
    return dict(
        ttm=0.25,
        phi_grid=np.array([0.0 + 0.0j, -0.5 + 0.7j]),
        sigma0=0.2,
        q=0.2,
        times=times,
        a0=np.tile(np.array([0.01, 0.02]), (times.size, 1)),
        a1=np.full(times.size, 0.015),
        kappa0=np.zeros(times.size),
        kappa1=np.full(times.size, 3.0),
        kappa2=np.full(times.size, 8.0),
        beta=np.tile(np.array([-0.2, 0.1]), (times.size, 1)),
        volvol=np.full(times.size, 0.3),
        b=np.full(times.size, 0.001),
        expansion_order=ExpansionOrder.FIRST,
    )


def _quadratic_inputs() -> dict:
    return dict(
        q=0.2,
        a0=np.array([0.01, 0.02]),
        a1=0.015,
        kappa0=0.0,
        kappa1=3.0,
        kappa2=8.0,
        beta=np.array([-0.2, 0.1]),
        volvol=0.3,
        b=0.001,
        phi=-0.5 + 0.7j,
        expansion_order=ExpansionOrder.FIRST,
    )


def test_factor_hjm_mgf_is_normalized_for_swap_and_futures_branches() -> None:
    results = {}
    for underlying in (UnderlyingType.SWAP, UnderlyingType.FUTURES):
        coefficients, log_mgf = compute_logsv_a_mgf_grid(
            underlying_type=underlying, **_mgf_inputs()
        )
        assert coefficients.shape == (2, 3)
        assert np.all(np.isfinite(coefficients))
        assert np.all(np.isfinite(log_mgf))
        np.testing.assert_allclose(log_mgf[0], 0.0, rtol=0.0, atol=1.0e-14)
        results[underlying] = log_mgf[1]

    assert results[UnderlyingType.SWAP] != results[UnderlyingType.FUTURES]


@pytest.mark.parametrize("underlying", [UnderlyingType.SWAP, UnderlyingType.FUTURES])
def test_factor_hjm_free_mgf_terms_match_direct_formula(
    underlying: UnderlyingType,
) -> None:
    inputs = _quadratic_inputs()
    matrices, linear, free = func_a_ode_quadratic_terms(
        underlying_type=underlying, **inputs
    )
    a_product = np.dot(inputs["a0"], inputs["a0"])
    if underlying == UnderlyingType.FUTURES:
        a_product += inputs["a1"] ** 2
    rhs = inputs["phi"] * (2.0 * inputs["b"] + a_product * inputs["phi"])
    expected = np.array(
        [0.5 * inputs["q"] ** 2 * rhs, inputs["q"] * rhs, 0.5 * rhs]
    )

    assert matrices.shape == (3, 3, 3)
    assert linear.shape == (3, 3)
    assert np.all(np.isfinite(matrices))
    assert np.all(np.isfinite(linear))
    np.testing.assert_allclose(free, expected, rtol=0.0, atol=1.0e-18)


def test_factor_hjm_rejects_unknown_underlying_precisely() -> None:
    with pytest.raises(NotImplementedError, match="underlying"):
        func_a_ode_quadratic_terms(underlying_type=object(), **_quadratic_inputs())


def test_factor_hjm_normal_volatility_uses_absolute_rate_units() -> None:
    """Market normal vols such as 150 bp must not be scaled by the rate forward."""
    forward = 0.04
    ttm = 1.0
    normal_vol = 0.015

    price = svm.compute_normal_price(
        forward=forward,
        strike=forward,
        ttm=ttm,
        vol=normal_vol,
        optiontype="C",
    )
    expected = normal_vol / np.sqrt(2.0 * np.pi)
    inferred = svm.infer_normal_implied_vol(
        forward=forward,
        strike=forward,
        ttm=ttm,
        given_price=price,
        optiontype="C",
    )

    np.testing.assert_allclose(price, expected, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(inferred, normal_vol, rtol=0.0, atol=1.0e-12)


def _swaption_params():
    from stochvolmodels.pricers.factor_hjm.rate_factor_basis import NelsonSiegel
    from stochvolmodels.pricers.factor_hjm.rate_logsv_params import (
        MultiFactRateLogSvParams,
        TermStructure,
    )

    times = np.array([0.0, 1.0, 2.0])
    return MultiFactRateLogSvParams(
        sigma0=1.0,
        theta=1.0,
        kappa1=0.25,
        kappa2=0.5,
        beta=TermStructure.create_multi_fact_from_vec(times, np.full(3, 0.2)),
        volvol=TermStructure.create_from_scalar(times, 0.2),
        A=np.full(3, 0.01),
        R=np.array([[1.0, 0.99, 0.97], [0.99, 1.0, 0.98], [0.97, 0.98, 1.0]]),
        basis=NelsonSiegel(meanrev=0.55, key_terms=np.array([2.0, 5.0, 10.0])),
        ccy="USD",
        vol_interpolation="BY_YIELD",
    )


def test_factor_hjm_annuity_measure_coefficients_are_finite() -> None:
    """The annuity-measure transform stores a scalar annuity per date (NumPy 2.5)."""
    from stochvolmodels.utils.rate_core import generate_ttms_grid

    params = _swaption_params()
    coefficients = params.transform_QA_params(
        expiry=1.0, tenor=2.0, t_grid=generate_ttms_grid(np.array([1.0]))
    )
    assert all(np.all(np.isfinite(values)) for values in coefficients[:6])
    assert params.check_QA_kappa2(expiry=1.0, tenor=2.0)


def test_factor_hjm_reduce_keeps_the_selected_expiries() -> None:
    params = _swaption_params()
    reduced = params.reduce(["2y"])
    np.testing.assert_array_equal(reduced.ts, [0.0, 2.0])
    np.testing.assert_array_equal(reduced.beta.xs, params.beta.xs[1:])


def test_factor_hjm_monte_carlo_inverts_each_tenor_at_its_forward() -> None:
    from stochvolmodels.pricers.factor_hjm.factor_hjm_pricer import calc_mc_vols

    params = _swaption_params()
    forwards = [np.array([0.0439]), np.array([0.0439])]
    strikes = [[np.array([0.04, 0.0439, 0.048])], [np.array([0.04, 0.0439, 0.048])]]
    _, mid, up, down = calc_mc_vols(
        basis_type="NELSON-SIEGEL",
        params=params,
        ttm=1.0,
        tenors=np.array([2.0, 5.0]),
        forwards=forwards,
        strikes_ttms=strikes,
        optiontypes=np.repeat("C", 3),
        is_annuity_measure=False,
        nb_path=200,
    )
    for low, centre, high in zip(down, mid, up):
        assert np.all(np.isfinite(centre)) and np.all(low <= centre) and np.all(centre <= high)


def test_factor_hjm_strikes_from_deltas_straddle_the_forward() -> None:
    """Before the fix a swallowed TypeError set every strike to the forward."""
    from stochvolmodels.pricers.factor_hjm.rate_logsv_ivols import infer_strikes_from_deltas

    strikes = infer_strikes_from_deltas(
        deltas=np.array([-0.25, 0.25]),
        f0=0.04,
        ttm=0.25,
        sigma0=0.01,
        rho=0.0,
        total_vol=0.5,
        beta=0.0,
        shift=0.0,
    ).values
    assert strikes[0] < 0.04 < strikes[1]
