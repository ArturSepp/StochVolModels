"""Independent numerical and information-set checks for dynamic-lambda ML."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import spx_lambda_ml as model

pytestmark = pytest.mark.repository_only


def synthetic():
    dates = pd.date_range("2000-01-01", periods=360, freq="MS", tz="UTC")
    t = np.arange(len(dates))
    raw = 3 + 4 * np.sin(t / 12) + 2 * np.cos(t / 55)
    slots = np.arange(48, 320, 2)
    variance = 0.08 + 0.04 * np.sin(slots / 20) ** 2
    means = np.column_stack([-0.5 * variance, -0.5 * variance])
    variances = np.column_stack([variance, variance])
    weights = np.tile(np.log([0.4, 0.6]), (len(slots), 1))
    lam = model.filtered_path(raw, 3.2, 0.035)[slots]
    x = -0.5 * variance + lam * variance
    sample = model.Sample(
        dates[slots] + pd.Timedelta(days=12),
        dates[slots] + pd.Timedelta(days=35),
        slots,
        x,
        weights,
        means,
        variances,
    )
    return dates, raw, sample


@pytest.mark.parametrize("half_life", [1.0, 2.0, 10.0])
@pytest.mark.parametrize("gain", [0.0, 2 / 121, 0.2, 1.0])
def test_fast_filter_equals_explicit_monthly_recursion(half_life, gain):
    _, raw, _ = synthetic()
    state, expected = 3.0, []
    for z in raw:
        prediction = 3 + 2 ** (-1 / (12 * half_life)) * (state - 3)
        state = prediction + gain * (z - prediction)
        expected.append(state)
    np.testing.assert_allclose(model.filtered_path(raw, half_life, gain), expected, atol=2e-13)


def test_likelihood_and_gradient_against_explicit_tilted_normal_densities():
    _, raw, sample = synthetic()
    pars = np.array([np.log(2.3), 0.027])
    got, gradient = model.nll_gradient(pars, raw, sample)
    lam = model.filtered_path(raw, np.exp(pars[0]), pars[1])[sample.indices]
    lw = sample.logw + lam[:, None] * sample.means + 0.5 * lam[:, None] ** 2 * sample.variances
    lp = lw - logsumexp(lw, axis=1)[:, None]
    p = logsumexp(
        lp
        + norm.logpdf(
            sample.x[:, None],
            sample.means + lam[:, None] * sample.variances,
            np.sqrt(sample.variances),
        ),
        axis=1,
    )
    q = logsumexp(
        sample.logw + norm.logpdf(sample.x[:, None], sample.means, np.sqrt(sample.variances)),
        axis=1,
    )
    assert got == pytest.approx(-np.sum(p - q), abs=1e-12)
    independent = []
    for i in range(2):
        step = np.zeros(2)
        step[i] = 1e-6
        independent.append(
            (
                model.nll_gradient(pars + step, raw, sample)[0]
                - model.nll_gradient(pars - step, raw, sample)[0]
            )
            / 2e-6
        )
    np.testing.assert_allclose(gradient, independent, rtol=2e-6, atol=1e-7)


def test_noise_free_population_objective_recovers_known_parameters():
    _, raw, sample = synthetic()
    estimate, profile = model.fit_parameters(raw, sample)
    assert estimate["half_life_years"] == pytest.approx(3.2, abs=0.02)
    assert estimate["gain"] == pytest.approx(0.035, abs=0.0001)
    benchmark = model.nll_gradient([np.log(2), 2 / 121], raw, sample)[0]
    assert estimate["negative_tilt_loglik"] <= benchmark + 1e-8
    assert profile.loglik_loss_from_joint.min() >= -1e-8


def test_zero_gain_makes_half_life_unidentified_not_an_estimated_two_year_speed():
    _, raw, sample = synthetic()
    sample.x = sample.means[:, 0] + 3 * sample.variances[:, 0]
    estimate, profile = model.fit_parameters(raw, sample)
    assert estimate["gain"] == 0
    assert not estimate["half_life_identified_at_gain"]
    assert estimate["half_life_years"] == 2  # Reporting convention only.
    assert estimate["profile_full_range_within_one_loglik"]
    assert np.ptp(profile.negative_tilt_loglik) < 1e-12


def test_same_timestamp_settlement_is_excluded():
    _, _, sample = synthetic()
    cutoff = sample.expiry[10]
    training = model.causal_training(sample, cutoff)
    assert len(training.x) == 10
    assert (training.expiry < cutoff).all()
    assert len(model.causal_training(sample, cutoff + pd.Timedelta(nanoseconds=1)).x) == 11


def test_walk_forward_fit_is_unchanged_by_future_returns_or_outcomes():
    dates, raw, sample = synthetic()
    cuts = dates[[80, 90]]
    original, profiles = model.calibration_history(dates, raw, sample, cuts, 10, 2, minimum=8)
    future_raw = raw.copy()
    future_raw[dates > cuts[-1]] = 90.0
    future_sample = sample.subset(np.ones(len(sample.x), dtype=bool))
    future_sample.x[future_sample.expiry >= cuts[-1]] += 100
    changed, new_profiles = model.calibration_history(
        dates, future_raw, future_sample, cuts, 10, 2, minimum=8
    )
    pd.testing.assert_frame_equal(original, changed)
    pd.testing.assert_frame_equal(profiles, new_profiles)
    assert (original.last_training_expiry < original.refit_date).all()
    assert original.half_life_years.between(1, 10).all()
    assert original.gain.between(0, 1).all()


def test_training_warmup_never_becomes_preferred_parameter_fallback():
    dates, raw, sample = synthetic()
    fits, profiles = model.calibration_history(dates, raw, sample, dates[[30, 40]], 10, 2)
    assert fits.status.eq("insufficient_training").all()
    assert "gain" not in fits and "half_life_years" not in fits
    assert profiles.empty


def test_stale_likelihood_sample_reuses_parameters_but_can_update_state():
    dates, raw, sample = synthetic()
    cuts = dates[[330, 331]]
    fits, _ = model.calibration_history(dates, raw, sample, cuts, 10, 2)
    assert fits.fit_id.nunique() == 1
    assert fits.parameters_fitted_at.eq(cuts[0]).all()
    np.testing.assert_array_equal(fits.gain, np.repeat(fits.gain.iloc[0], 2))
    values = model.filtered_path(raw, fits.half_life_years.iloc[0], fits.gain.iloc[0])
    assert values[330] != values[331]


@pytest.mark.parametrize("half_life,gain", [(0.99, 0.1), (10.01, 0.1), (2, -0.01), (2, 1.01)])
def test_parameter_bounds_are_enforced(half_life, gain):
    with pytest.raises(ValueError, match="Half-life"):
        model.filtered_path(np.array([3.0, 4.0]), half_life, gain)


def test_missing_roots_are_explicit_failures():
    with pytest.raises(ValueError, match="finite"):
        model.filtered_path(np.array([3.0, np.nan]), 2, 0.01)
