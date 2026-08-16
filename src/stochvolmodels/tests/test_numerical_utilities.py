import numpy as np
import pytest
from stochvolmodels.utils.config import VariableType

from stochvolmodels.utils.mc_payoffs import compute_mc_vars_payoff
from stochvolmodels.utils.mgf_pricer import compute_integration_weights


def test_simpson_weights_integrate_low_order_polynomials() -> None:
    integration_grid = np.linspace(-2.0, 2.0, 9)
    transform_grid = 0.5 + 1j * integration_grid
    weights = compute_integration_weights(transform_grid, is_simpson=True)

    np.testing.assert_allclose(np.dot(weights, np.ones(9)), 4.0, atol=1.0e-14)
    np.testing.assert_allclose(np.dot(weights, integration_grid), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        np.dot(weights, integration_grid**2),
        16.0 / 3.0,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(np.dot(weights, integration_grid**3), 0.0, atol=1.0e-14)


def test_trapezoidal_weights_integrate_constant_and_linear_functions() -> None:
    integration_grid = np.linspace(0.0, 1.0, 5)
    transform_grid = -0.5 + 1j * integration_grid
    weights = compute_integration_weights(transform_grid, is_simpson=False)

    np.testing.assert_allclose(np.dot(weights, np.ones(5)), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(np.dot(weights, integration_grid), 0.5, atol=1.0e-14)


def test_simpson_weights_reject_even_point_grid() -> None:
    transform_grid = 1j * np.linspace(0.0, 1.0, 4)
    with pytest.raises(ValueError, match="odd"):
        compute_integration_weights(transform_grid, is_simpson=True)


@pytest.mark.parametrize(
    ("grid", "is_simpson", "message"),
    [
        (np.array([0.0]), False, "too short"),
        (np.array([0.0, 0.5, 1.1]), True, "uniform"),
        (np.array([0.0, 0.5, 0.4]), True, "increasing"),
        (np.array([0.0, np.nan, 1.0]), True, "finite"),
    ],
)
def test_integration_weights_reject_invalid_grids(
    grid: np.ndarray, is_simpson: bool, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_integration_weights(1j * grid, is_simpson=is_simpson)


def _expected_payoffs(spots: np.ndarray, strike: float) -> np.ndarray:
    return np.vstack(
        [
            np.maximum(spots - strike, 0.0),
            np.maximum(strike - spots, 0.0),
            np.maximum(spots - strike, 0.0) / spots,
            np.maximum(strike - spots, 0.0) / spots,
        ]
    )


def test_mc_payoff_codes_match_direct_pathwise_calculation() -> None:
    spots = np.array([0.8, 1.0, 1.2])
    x0 = np.log(spots)
    strikes = np.ones(4)
    optiontypes = np.array(["C", "P", "IC", "IP"])
    discfactor = 0.95

    prices, standard_errors = compute_mc_vars_payoff(
        x0=x0,
        sigma0=np.ones_like(x0),
        qvar0=np.zeros_like(x0),
        ttm=1.0,
        forward=1.0,
        strikes_ttm=strikes,
        optiontypes_ttm=optiontypes,
        discfactor=discfactor,
        variable_type=VariableType.LOG_RETURN,
    )
    payoffs = _expected_payoffs(spots, strike=1.0)

    np.testing.assert_allclose(prices, discfactor * np.mean(payoffs, axis=1), atol=1.0e-14)
    np.testing.assert_allclose(
        standard_errors,
        discfactor * np.std(payoffs, axis=1) / np.sqrt(spots.size),
        atol=1.0e-14,
    )


def test_mc_standard_error_scales_with_inverse_sqrt_path_count() -> None:
    x0 = np.log(np.array([0.75, 0.95, 1.05, 1.25]))
    strikes = np.array([1.0])
    optiontypes = np.array(["C"])
    common = dict(
        ttm=1.0,
        forward=1.0,
        strikes_ttm=strikes,
        optiontypes_ttm=optiontypes,
        variable_type=VariableType.LOG_RETURN,
    )

    prices, standard_errors = compute_mc_vars_payoff(
        x0=x0,
        sigma0=np.ones_like(x0),
        qvar0=np.zeros_like(x0),
        **common,
    )
    repeated_x0 = np.tile(x0, 4)
    repeated_prices, repeated_errors = compute_mc_vars_payoff(
        x0=repeated_x0,
        sigma0=np.ones_like(repeated_x0),
        qvar0=np.zeros_like(repeated_x0),
        **common,
    )

    np.testing.assert_allclose(repeated_prices, prices, atol=1.0e-14)
    np.testing.assert_allclose(repeated_errors, standard_errors / 2.0, atol=1.0e-14)


def test_mc_qvar_payoff_uses_annualized_quadratic_variance() -> None:
    qvar0 = np.array([0.02, 0.08, 0.18])
    ttm = 0.5
    strikes = np.array([0.15, 0.15])
    optiontypes = np.array(["C", "P"])

    prices, _ = compute_mc_vars_payoff(
        x0=np.zeros(qvar0.size),
        sigma0=np.ones(qvar0.size),
        qvar0=qvar0,
        ttm=ttm,
        forward=1.0,
        strikes_ttm=strikes,
        optiontypes_ttm=optiontypes,
        variable_type=VariableType.Q_VAR,
    )
    underlying = qvar0 / ttm
    expected = np.array(
        [
            np.mean(np.maximum(underlying - strikes[0], 0.0)),
            np.mean(np.maximum(strikes[1] - underlying, 0.0)),
        ]
    )
    np.testing.assert_allclose(prices, expected, atol=1.0e-14)


def test_mc_payoff_rejects_unknown_option_type() -> None:
    with pytest.raises(ValueError, match="payoff"):
        compute_mc_vars_payoff(
            x0=np.zeros(4),
            sigma0=np.ones(4),
            qvar0=np.zeros(4),
            ttm=1.0,
            forward=1.0,
            strikes_ttm=np.array([1.0]),
            optiontypes_ttm=np.array(["BAD"]),
        )
