"""Independent quadrature, premium and failure checks for the shared vanilla integrator."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import roots_legendre
import vanilla_option_pricers as vop

from stochvolmodels.utils import _fourier_quadrature as fq


def test_mapped_rule_integrates_polynomials_on_offset_panels() -> None:
    """Check analytic moments and independently locate all seven embedded weights."""
    edges = np.array([0.0, 0.75, 2.25, 6.0])
    plan = fq.FourierPlan(edges)
    gauss, gauss_weights = roots_legendre(7)
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        panel = slice(15*i, 15*(i+1))
        u = plan.nodes[panel]
        fine = plan.weights[panel]
        coarse = plan.embedded_weights[panel]
        active = coarse > 0
        half_width = 0.5 * (right - left)
        np.testing.assert_allclose(u[active], left + half_width*(1 + gauss), atol=2e-15, rtol=0)
        np.testing.assert_allclose(coarse[active], half_width*gauss_weights, atol=5e-15, rtol=0)
        for degree in (0, 1, 2, 6, 12):
            exact = (right**(degree+1) - left**(degree+1)) / (degree+1)
            np.testing.assert_allclose([fine @ u**degree, coarse @ u**degree], exact, rtol=3e-14)
    edges[1] = 0.5
    assert plan.edges[1] == 0.75
    for array in (plan.edges, plan.nodes, plan.weights, plan.embedded_weights):
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("is_spot", [True, False], ids=["money-market", "inverse"])
def test_discounted_mixed_vanillas_and_sample_reuse(is_spot: bool) -> None:
    """A martingale mixture of two Black forwards has known prices and a complex transform."""
    ttm, forward, discount, vol = 0.4, 112.0, 0.957, 0.21
    strikes = np.array([87.0, 110.0, 140.0, 170.0, 110.0])
    types = np.array(["P", "C", "C", "P", "P"])
    inverse_types = np.array(["IP", "IC", "C", "P", "IP"], dtype=object)
    # The multipliers have probability-weighted mean one. The change of numeraire
    # gives M_inverse(phi) = M_spot(phi - 1) for the normalized forward.
    observed = []

    def log_mgf(phi):
        observed.append(phi.copy())
        z = phi if is_spot else phi - 1.0
        mixture = 0.8 * np.exp(-z*np.log(0.9)) + 0.2 * np.exp(-z*np.log(1.4))
        return 0.5 * vol**2 * ttm * z*(z + 1.0) + np.log(mixture)

    plan, prices, diag = fq.integrate_vanilla_slice(
        log_mgf, maturity=ttm, forward=forward, strikes=strikes,
        optiontypes=types if is_spot else inverse_types, discfactor=discount,
        is_spot_measure=is_spot, price_atol=1e-7,
    )
    expected = sum(
        prob * vop.compute_bsm_vanilla_slice_prices(
            ttm, forward*multiplier, strikes, np.full(strikes.shape, vol), types, discount,
        )
        for prob, multiplier in [(0.8, 0.9), (0.2, 1.4)]
    )
    np.testing.assert_allclose(prices, expected, atol=1e-7, rtol=0)
    evaluated = np.concatenate(observed)
    np.testing.assert_array_equal(evaluated.real, -0.5 if is_spot else 0.5)
    assert diag.evaluations == evaluated.size == np.unique(evaluated.imag).size
    assert np.isin(plan.nodes, evaluated.imag).all()
    assert diag.refinements > 0
    assert diag.evaluations == plan.nodes.size + 15*diag.refinements
    assert diag.panels == len(plan.edges) - 1
    assert diag.cutoff == plan.edges[-1]
    assert max(diag.resolution_error, diag.tail_estimate) <= 0.5e-7


def _black_log_mgf(phi: np.ndarray) -> np.ndarray:
    return 0.5 * 0.2**2 * phi*(phi + 1.0)


def _atm_slice(log_mgf=_black_log_mgf, **kwargs):
    return fq.integrate_vanilla_slice(
        log_mgf, maturity=1.0, forward=100.0, strikes=np.array([100.0]),
        optiontypes=np.array(["C"]), **kwargs,
    )


def test_frozen_plan_reprices_new_samples() -> None:
    """A frozen plan accepts fresh model values and surfaces invalid samples."""
    plan, first, _ = _atm_slice()
    phi = -0.5 + 1j * plan.nodes
    vol = 0.25
    samples = 0.5 * vol**2 * phi*(phi + 1.0)
    quotes = dict(forward=100.0, strikes=np.array([100.0]), optiontypes=np.array(["C"]))
    second = plan.price_vanillas(samples, **quotes)
    expected = vop.compute_bsm_vanilla_slice_prices(
        1.0, 100.0, quotes["strikes"], np.array([vol]), quotes["optiontypes"], 1.0,
    )
    np.testing.assert_allclose(second, expected, atol=1e-6, rtol=0)
    assert second[0] > first[0]
    with pytest.raises(ValueError, match="one log transform"):
        plan.price_vanillas(samples[:-1], **quotes)
    samples[-1] = np.nan
    with pytest.raises(FloatingPointError, match="nonfinite log transform"):
        plan.price_vanillas(samples, **quotes)


def test_diagnostics_match_final_plan() -> None:
    """Reconstruct premium errors from fixed final nodes with damping and strike units."""
    strike, discount, vol = 190.0, 0.93, 0.035
    seen = []

    def log_mgf(phi):
        seen.append(phi.size)
        return 0.5 * vol**2 * phi*(phi + 1.0)

    plan, _, diag = fq.integrate_vanilla_slice(
        log_mgf, maturity=1.0, forward=100.0, strikes=np.array([strike]),
        optiontypes=np.array(["C"]), discfactor=discount, price_atol=1e-7,
    )
    u = plan.nodes
    scale = discount*np.sqrt(100.0*strike)/np.pi
    amplitude = scale*np.exp(-0.5*vol**2*(u*u + 0.25))/(u*u + 0.25)
    signed = amplitude*np.cos(np.log(100.0/strike)*u)
    diffs = (signed * (plan.weights - plan.embedded_weights)).reshape(-1, 15).sum(axis=1)
    np.testing.assert_allclose(diag.resolution_error, np.abs(diffs).sum(), atol=2e-14, rtol=0)
    assert diag.refinements > 0 and sum(seen) == diag.evaluations
    last_two = []
    for lo, hi in [(diag.cutoff/4, diag.cutoff/2), (diag.cutoff/2, diag.cutoff)]:
        inside = (u > lo) & (u < hi)
        last_two.append(plan.weights[inside] @ amplitude[inside])
    np.testing.assert_allclose(diag.tail_estimate, max(last_two), atol=0.0, rtol=1e-12)


def test_outer_shell_mass_survives_panel_bisections() -> None:
    """Bisection improves resolution and retains the full two-shell tail mass."""
    seen = []
    vol, forward, strike, discount = 0.035, 100.0, 190.0, 0.93

    def model(phi):
        seen.extend(phi.imag)
        return 0.5 * vol**2 * phi*(phi + 1.0)

    with pytest.raises(fq.QuadratureError, match="cutoff budget") as failure:
        fq.integrate_vanilla_slice(
            model, maturity=1.0, forward=forward, strikes=np.array([strike]),
            optiontypes=np.array(["C"]), discfactor=discount, price_atol=1e-9, max_cutoff=64.0,
        )
    # These centers are evaluated only after both [16,32] and [32,64] are split.
    assert np.isin([20.0, 28.0, 40.0, 56.0], seen).all()

    def envelope(u):
        return np.exp(-0.5 * vol**2 * (u*u + 0.25)) / (u*u + 0.25)

    larger_shell = quad(envelope, 16.0, 32.0)[0]
    expected_mass = discount * np.sqrt(forward*strike)/np.pi * larger_shell
    assert failure.value.diagnostics.tail_estimate == pytest.approx(expected_mass, rel=1e-12)


def test_scrambled_embedded_weights_reject_unresolved_panel(monkeypatch) -> None:
    """A permutation preserves total weight but corrupts the coarse error estimate."""
    # This loose tolerance genuinely resolves the first panel without bisection.
    _, _, clean = _atm_slice(price_atol=0.01, max_refinements=0)
    assert clean.refinements == 0
    monkeypatch.setattr(fq, "_EMBEDDED", np.roll(fq._EMBEDDED, 1))
    with pytest.raises(fq.QuadratureError, match="refinement budget") as rejected:
        _atm_slice(price_atol=0.01, max_refinements=0)
    assert rejected.value.diagnostics.resolution_error > 0.005


def test_truncated_slow_tail_fails_despite_resolved_panels() -> None:
    """A deterministic ATM forward has unit transform and an algebraically decaying tail."""
    with pytest.raises(fq.QuadratureError, match="cutoff budget") as failure:
        _atm_slice(np.zeros_like, initial_cutoff=1.0, max_cutoff=16.0, price_atol=1e-6)
    diag = failure.value.diagnostics
    assert (diag.maturity, diag.strike, diag.cutoff) == (1.0, 100.0, 16.0)
    assert diag.resolution_error <= 0.5e-6
    # Exact integral of 100 / [pi*(u^2+1/4)] over [4,8], the larger outer shell.
    exact = 200.0/np.pi * (np.arctan(16.0) - np.arctan(8.0))
    assert diag.tail_estimate == pytest.approx(exact, abs=1e-10, rel=0)


def test_exhausted_frequency_budget_does_not_call_model_again() -> None:
    observed = []

    def model(phi):
        observed.append(phi.size)
        return _black_log_mgf(phi)

    with pytest.raises(fq.QuadratureError, match="frequency evaluation budget") as failure:
        _atm_slice(model, max_nodes=15)
    assert observed == [15]
    assert failure.value.diagnostics.evaluations == 15


@pytest.mark.parametrize("value", [complex(np.nan), complex(0, np.inf), 900+0j])
def test_invalid_model_samples_fail_before_price_reduction(value) -> None:
    def model(phi):
        values = _black_log_mgf(phi)
        values[-1] = value
        return values

    with pytest.raises(fq.QuadratureError) as failure:
        _atm_slice(model)
    assert failure.value.diagnostics.evaluations == 15


def test_combined_tolerance_matches_explicit_quote_budgets() -> None:
    kwargs = dict(maturity=1.0, forward=100., strikes=np.array([85., 110.]),
                  optiontypes=np.array(["P", "C"]))
    time_values = np.array([0.1, 3.0])
    atol = np.array([1e-7, 5e-7])
    combined, prices, _ = fq.integrate_vanilla_slice(
        _black_log_mgf, **kwargs, price_atol=atol, price_rtol=1e-4, time_values=time_values,
    )
    explicit, reference, _ = fq.integrate_vanilla_slice(
        _black_log_mgf, **kwargs, price_atol=atol + 1e-4*time_values,
    )
    np.testing.assert_array_equal(combined.edges, explicit.edges)
    np.testing.assert_array_equal(prices, reference)


@pytest.mark.parametrize("options", [
    {"price_atol": 0.0}, {"price_rtol": 1e-3}, {"max_nodes": True},
    {"max_refinements": -1}, {"initial_cutoff": 2.0, "max_cutoff": 1.0},
])
def test_invalid_controls_rejected_before_model_evaluation(options) -> None:
    with pytest.raises(ValueError):
        _atm_slice(lambda phi: pytest.fail("unexpected transform evaluation"), **options)
