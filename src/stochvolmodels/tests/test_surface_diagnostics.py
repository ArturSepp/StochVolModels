"""Surface checks against analytic Black references and deliberate defects."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import ndtr

from stochvolmodels.fitters.surface_checks import (
    SurfaceEvaluator,
    check_calendar,
    check_call_slice,
    diagnose_surface,
)


def reference(strikes, forward=100.0, discount=1.02, ttm=0.5, vol=0.3):
    """Black prices and density evaluated independently of the production pricer."""
    scale = vol * np.sqrt(ttm)
    d2 = np.log(forward / strikes) / scale - scale / 2
    prices = discount * (forward * ndtr(d2 + scale) - strikes * ndtr(d2))
    density = np.exp(-(d2**2) / 2) / (np.sqrt(2 * np.pi) * strikes * scale)
    return prices, density


def evaluator(iv=None):
    """One underlying with nonflat forwards and negative rates."""
    return SurfaceEvaluator(
        iv=iv or (lambda strikes, t: np.full_like(strikes, 0.3)),
        forward=lambda t: 100 * np.exp(0.08 * t),
        discount=lambda t: np.exp(0.02 * t),
        ttm_domain=(0.1, 1.0),
        k_domain=(-0.6, 0.6),
        underlying_id="synthetic equity",
        interpolation="analytic",
        carry_model="proportional",
    )


def test_black_reference_and_negative_rates_pass_on_irregular_strikes():
    strikes = np.array([60.0, 72.0, 88.0, 94.0, 100.0, 108.0, 120.0, 145.0, 180.0])
    prices, density = reference(strikes)
    report = check_call_slice(
        strikes, prices, density=density, forward=100, discount=1.02, price_unit="USD per unit"
    )
    assert set(report.table("checks").status) == {"pass"}
    assert report.metadata["negative_density_mass_on_grid"] == 0
    assert report.metadata["density_measure"] == "dK"
    assert report.metadata["global_certificate"] is False


def test_nonconvex_curve_fails_at_known_strike():
    report = check_call_slice(
        [90, 100, 120], [12, 10, 1], forward=100, discount=1, price_unit="USD"
    )
    checks = report.table("checks").set_index("check_id")
    assert checks.loc["convexity", "status"] == "fail"
    assert checks.loc["convexity", "location"] == 100
    assert checks.loc["convexity", "value"] == pytest.approx(-0.25)
    assert checks.loc["density_nonnegative", "value"] == pytest.approx(-1 / 60)


def test_grid_density_converges_to_independent_analytic_reference():
    errors = []
    for n in (101, 201):
        strikes = np.linspace(60, 150, n)
        prices, density = reference(strikes)
        report = check_call_slice(strikes, prices, forward=100, discount=1.02, price_unit="USD")
        errors.append(np.max(np.abs(report.table("density").density - density[1:-1])))
    assert errors[1] < errors[0] / 3
    assert errors[1] < 1e-6


def test_partial_and_short_grids_cannot_pass_every_check():
    report = check_call_slice(
        [90, 100, 120], [12, np.nan, 1], forward=100, discount=1, price_unit="USD"
    )
    checks = report.table("checks")
    assert set(checks.status) == {"not_checked"}
    tiny = check_call_slice([100], [1], forward=100, discount=1, price_unit="USD")
    assert tiny.table("checks").set_index("check_id").loc["convexity", "status"] == "not_checked"


def test_call_bounds_and_vertical_spread_failures():
    bounds = check_call_slice(
        [90, 100, 120], [5, 110, 1], forward=100, discount=1, price_unit="USD"
    )
    assert bounds.table("checks").set_index("check_id").loc["call_bounds", "status"] == "fail"
    vertical = check_call_slice(
        [90, 100, 120], [12, 13, 1], forward=100, discount=1, price_unit="USD"
    )
    assert vertical.table("checks").set_index("check_id").loc["vertical_spread", "status"] == "fail"


def test_density_mass_is_not_normalized_and_nan_intervals_are_not_bridged():
    report = check_call_slice(
        [80, 90, 100, 110, 120],
        [21, 13, 7, 3, 1],
        forward=100,
        discount=1,
        price_unit="USD",
        density=[-0.01, -0.01, np.nan, 0.01, 0.01],
    )
    assert report.metadata["negative_density_mass_on_grid"] == pytest.approx(0.1)
    assert report.metadata["density_mass_on_grid"] == pytest.approx(0)
    assert report.metadata["density_integrated_intervals"] == 2
    assert (
        report.table("checks").set_index("check_id").loc["density_nonnegative", "status"] == "fail"
    )


def test_calendar_detects_crossing_and_requires_declared_assumptions():
    variance = np.array([[0.05, 0.04, 0.05], [0.06, 0.03, 0.07]])
    kwargs = dict(underlying_id="same asset", carry_model="proportional")
    check = check_calendar([0.2, 0.4], [-0.2, 0, 0.2], variance, **kwargs).table("checks").iloc[0]
    assert check.status == "fail"
    assert check.location == 0
    assert check.value == pytest.approx(-0.01)
    assert check.next_ttm == 0.4
    unspecified = check_calendar([0.2, 0.4], [-0.2, 0, 0.2], variance)
    assert unspecified.table("checks").status.iloc[0] == "not_checked"
    unsupported = check_calendar(
        [0.2, 0.4],
        [-0.2, 0, 0.2],
        variance,
        underlying_id="VIX futures",
        carry_model="different_underlyings",
    )
    assert unsupported.table("checks").status.iloc[0] == "not_applicable"


def test_declining_vol_is_not_declining_total_variance():
    variance = np.array([[0.2 * 0.4**2] * 3, [0.4 * 0.3**2] * 3])
    report = check_calendar(
        [0.2, 0.4], [-0.2, 0, 0.2], variance, underlying_id="same asset", carry_model="proportional"
    )
    assert report.table("checks").status.iloc[0] == "pass"
    variance[1, 0] = np.nan
    incomplete = check_calendar(
        [0.2, 0.4], [-0.2, 0, 0.2], variance, underlying_id="same asset", carry_model="proportional"
    )
    assert incomplete.table("checks").status.iloc[0] == "not_checked"


def test_surface_uses_each_forward_and_matches_reference_prices():
    report = diagnose_surface(
        evaluator(), [0.2, 0.6], [-0.2, 0, 0.2], price_unit="USD", extra_ttms=[0.4]
    )
    assert set(report.table("checks").status) == {"pass"}
    for t, group in report.table("grid").groupby("ttm"):
        forward, discount = 100 * np.exp(0.08 * t), np.exp(0.02 * t)
        np.testing.assert_allclose(group.strike, forward * np.exp([-0.2, 0, 0.2]))
        expected, _ = reference(group.strike.to_numpy(), forward=forward, discount=discount, ttm=t)
        np.testing.assert_allclose(group.call_price, expected, rtol=1e-12, atol=1e-12)


def test_intermediate_maturity_can_fail_despite_good_endpoints():
    def bad_iv(strikes, t):
        return np.full_like(strikes, 0.05 if 0.3 < t < 0.5 else 0.3)

    model = evaluator(bad_iv)
    endpoints = diagnose_surface(model, [0.2, 0.6], [-0.2, 0, 0.2], price_unit="USD")
    checks = endpoints.table("checks")
    assert checks.loc[checks.check_id == "calendar", "status"].iloc[0] == "pass"
    assert (
        checks.loc[checks.check_id == "intermediate_maturities", "status"].iloc[0] == "not_checked"
    )
    full = diagnose_surface(model, [0.2, 0.6], [-0.2, 0, 0.2], extra_ttms=[0.4], price_unit="USD")
    checks = full.table("checks")
    assert "fail" in checks.loc[checks.check_id == "calendar", "status"].tolist()


def test_domains_and_slice_only_interpolation_are_explicit():
    with pytest.raises(ValueError, match="extrapolation"):
        diagnose_surface(evaluator(), [0.2, 0.6], [-0.7, 0, 0.7], price_unit="USD")
    with pytest.raises(ValueError, match="interpolation"):
        diagnose_surface(
            replace(evaluator(), interpolation="none"),
            [0.2, 0.6],
            [-0.2, 0, 0.2],
            extra_ttms=[0.4],
            price_unit="USD",
        )
    report = diagnose_surface(
        replace(evaluator(), extrapolation="analytic"),
        [0.2, 0.6],
        [-0.7, 0, 0.7],
        allow_extrapolation=True,
        price_unit="USD",
    )
    assert report.table("grid").extrapolated.sum() == 4
    with pytest.raises(ValueError, match="k_domain"):
        diagnose_surface(
            replace(evaluator(), k_domain=(0.2, -0.2)), [0.2, 0.6], [-0.2, 0, 0.2], price_unit="USD"
        )


def test_unsupported_payoff_is_not_priced_as_a_standard_call():
    def forbidden(strikes, t):
        raise AssertionError("unsupported payoff must not be evaluated")

    report = diagnose_surface(
        replace(evaluator(forbidden), payoff="inverse"),
        [0.2, 0.6],
        [-0.2, 0, 0.2],
        price_unit="coin",
    )
    assert report.table("checks").status.iloc[0] == "not_applicable"


@pytest.mark.parametrize("vol", [0.0, -1.0, np.nan, np.inf])
def test_invalid_vols_cannot_become_valid_by_squaring_or_clipping(vol):
    report = diagnose_surface(
        evaluator(lambda strikes, t: np.full_like(strikes, vol)),
        [0.2, 0.6],
        [-0.2, 0, 0.2],
        price_unit="USD",
    )
    assert set(report.table("checks").status) == {"not_checked"}
