"""Independent controls for the repository's transient volatility reference."""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.repository_only


@pytest.fixture
def study(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    if not (root / "validation").is_dir():
        pytest.skip("requires the PDE validation checkout")
    monkeypatch.syspath_prepend(str(root))
    from validation import pde_volatility_moments
    return pde_volatility_moments


def test_log_flux_and_geometric_diffusion(study):
    from stochvolmodels import LogSvParams

    params = LogSvParams(sigma0=.2, theta=.35, kappa1=0., kappa2=0., beta=-.3, volvol=.6)
    y = np.linspace(-4., 1., 33)
    edges, generator = study._log_generator(params, y)
    # Zero-flux stationary law in log coordinates has density proportional to exp(-y).
    stationary = np.diff(edges) * np.exp(-y)
    np.testing.assert_allclose(generator @ stationary, 0., atol=4e-13)
    np.testing.assert_allclose(generator.toarray().sum(axis=0), 0., atol=4e-13)
    assert generator[1, 0] > 0 and generator[-2, -1] > 0

    res = study.Resolution(lower=12, upper=8, cells_per_octave=64, steps=512)
    run = study._measure("log", params, res)
    assert run["density_pass"]
    errors = study._errors(run["checkpoints"], study._analytic(params))
    assert errors["passed"], errors


def test_linear_mean_and_second_moment(study):
    from stochvolmodels import LogSvParams

    params = LogSvParams(sigma0=.2, theta=.35, kappa1=1.3, kappa2=0., beta=-.3, volvol=.6)
    # Integrating-factor solution gives an independent check on the moment-system exponential.
    t = np.array(study.TIMES)
    k, theta, s0 = params.kappa1, params.theta, params.sigma0
    v = params.beta**2 + params.volvol**2 - 2*k
    mean = theta + (s0 - theta)*np.exp(-k*t)
    second = s0**2*np.exp(v*t) + 2*k*theta * (
        theta*np.expm1(v*t)/v + (s0-theta)*(np.exp(v*t)-np.exp(-k*t))/(v+k))
    run = study._measure("log", params, study.Resolution(cells_per_octave=64, steps=512))
    for key, truth in (("mean", mean), ("second_moment", second), ("variance", second-mean**2)):
        np.testing.assert_allclose([r[key] for r in run["checkpoints"]], truth, rtol=1e-3, atol=0)


def test_cell_mass_cdf_and_right_tail(study):
    levels = np.array([0.25, 0.75, 1.25])
    edges = np.array([0., .5, 1., 1.5])
    mass = np.array([.25, .75 - 1e-18, 1e-18])
    result = study._observations(levels, edges, mass, logarithmic=False)
    # A tiny occupied tail must survive cancellation. The cut at 0.2 splits the first cell.
    i = np.flatnonzero(study.TAIL_CUTS == 1.)[0]
    assert result["upper_tail_second_moment"][i] == pytest.approx(1.25**2 * 1e-18, abs=1e-30)
    j = np.flatnonzero(study.CDF_CUTS == .2)[0]
    assert result["cdf"][j] == pytest.approx(.1)
    obs = study._observations(levels, np.log([.125, .5, 1., 1.5]), mass, logarithmic=True)
    assert obs["cdf"][j] == pytest.approx(.25 * np.log(.2/.125) / np.log(.5/.125))
