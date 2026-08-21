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
