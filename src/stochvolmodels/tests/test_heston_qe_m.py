"""CIR transition, uncentered martingale, public payoff and Euler replay checks."""

import numpy as np
import pytest
from scipy.special import ndtri, roots_hermitenorm, roots_laguerre
from scipy.stats import ncx2

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.heston import HestonModel, HestonParams
from stochvolmodels.models.heston.kernels import _qe_m_step
from stochvolmodels.pricers.heston_pricer import HestonPricer
from stochvolmodels.tests.test_fourier_reference_cases import (
    HESTON_STRESS,
    _heston_probability_calls,
)
from stochvolmodels.utils.config import VariableType
from stochvolmodels.utils.funcs import set_seed


def _request(**changes):
    return dict(
        dict(
            measure="Q_MMA",
            observation_times=np.array([0.0, 0.01]),
            spot0=1.0,
            n_paths=4,
            steps_per_year=360,
            seed=19,
        ),
        **changes,
    )


def test_historical_replay_for_model_terminal_and_chain():
    params = HestonParams(0.001, 0.04, 0.5, -0.9, 1.0)
    request = _request()
    model = HestonModel(params)
    pricer = HestonPricer()
    chain = OptionChain.slice_to_chain(0.01, 100.0, np.array([100.0]), np.array(["C"]), 0.97)
    expected = np.array(
        [
            [
                -0.00020253918811764056,
                -0.0076575586120942634,
                -0.0012421926674251454,
                0.000010528215511884688,
            ],
            [0.00433600282079423, 0.00685246975691437, 0.00835621863582145, 0.00195547424935281],
            [
                2.5838188405796822e-5,
                2.1028534593637092e-5,
                2.6023703279171449e-5,
                1.1061747307647116e-5,
            ],
        ]
    )
    replays = []
    for options in ({}, {"mc_scheme": "euler"}):
        path = model.simulate_paths(**request, **options)
        states = [
            path.states[key][:, -1]
            for key in ("log_zero_drift_return", "variance", "quadratic_variance")
        ]
        set_seed(19)
        terminal = pricer.simulate_terminal_values(params, ttm=0.01, nb_path=4, **options)
        set_seed(19)
        px, se = pricer.model_mc_price_chain(chain, params, nb_path=4, **options)
        np.testing.assert_allclose(states, expected, atol=1e-15, rtol=1e-12)
        np.testing.assert_allclose(terminal, expected, atol=1e-15, rtol=1e-12)
        np.testing.assert_allclose(
            [px[0][0], se[0][0]], [0.13004990511760853, 0.04401272302613813], atol=1e-12, rtol=0
        )
        replays.append(np.r_[np.ravel(states), np.ravel(terminal), px[0], se[0]])
    np.testing.assert_array_equal(*replays)


@pytest.mark.parametrize("psi", [0.4, 1.49999, 1.50001, 6.0])
@pytest.mark.parametrize("rho", [-1.0, 0.7, 1.0])
@pytest.mark.parametrize("kappa", [0.0, 2.0])
def test_one_step_moments_and_raw_forward_by_quadrature(psi, rho, kappa):
    xi, dt = 0.8, 0.2
    v = xi**2 * dt / psi
    mean = v
    if kappa:
        # Exact noncentral chi-square transition at unit vol-of-vol. Its variance
        # scales with xi squared and the mean is independent of xi.
        scale = -np.expm1(-kappa * dt) / (4 * kappa)
        mean, var_at_unit_xi = ncx2.stats(4 * kappa * .04, v * np.exp(-kappa*dt) / scale,
                                           scale=scale, moments="mv")
        xi = np.sqrt(psi / var_at_unit_xi) * mean
    if psi <= 1.5:
        z, w = roots_hermitenorm(56)
        w /= np.sqrt(2 * np.pi)
        atom = 0.0
    else:
        tail = 2 / (1 + psi)
        atom = 1 - tail
        e, weights = roots_laguerre(56)
        z = np.r_[ndtri(atom / 2), -ndtri(tail * np.exp(-e))]
        w = np.r_[atom, tail * weights]
    variance, x, qv = np.full_like(z, v), np.zeros_like(z), np.zeros_like(z)
    _qe_m_step(x, variance, qv, 0.04, kappa, rho, xi, dt, z, np.zeros_like(z))
    np.testing.assert_allclose(w @ variance, mean, rtol=2e-12)
    np.testing.assert_allclose(w @ (variance - mean) ** 2, psi * mean**2, rtol=2e-12)
    assert w[variance == 0.0].sum() == pytest.approx(atom)
    np.testing.assert_allclose(qv, dt * (v + variance) / 2)
    # Integrate the independent stock normal analytically before the variance quadrature.
    assert w @ np.exp(x + 0.5 * (1 - rho**2) * qv) == pytest.approx(1.0, abs=2e-12, rel=0)


@pytest.mark.parametrize(("v", "branch"), [(20.0, "mixture"), (100.0, "quadratic")])
def test_nonexistent_conditional_moment(v, branch):
    with pytest.raises(FloatingPointError, match=branch):
        _qe_m_step(
            np.zeros(1),
            np.array([v]),
            np.zeros(1),
            0.04,
            3.0,
            1.0,
            4.0,
            1.0,
            np.zeros(1),
            np.zeros(1),
        )


@pytest.mark.parametrize("xi", [0.0, 1e-14])
def test_deterministic_variance_black_limit(xi):
    p = HestonParams(0.09, 0.04, 1.0, -0.95, xi)
    chain = OptionChain.slice_to_chain(0.5, 1.0, np.array([1.0]), np.array(["C"]))
    price, se = HestonPricer().model_mc_price_chain(
        chain, p, nb_path=32000, mc_scheme="qe_m", seed=917, steps_per_year=48
    )
    from scipy.special import ndtr

    ivar = 0.04 * 0.5 + 0.05 * (-np.expm1(-0.5))
    expected = 2 * ndtr(np.sqrt(ivar) / 2) - 1
    assert se[0][0] < 0.001 and abs(price[0][0] - expected) < 4 * se[0][0]


def test_public_routes_match_the_raw_payoff_and_se():
    p = HestonParams(0.0001, 0.04, 1.0, 0.5, 0.6)
    times = np.array([0.0, 0.1, 0.25])
    paths = HestonModel(p).simulate_paths(
        **_request(observation_times=times, n_paths=4096, steps_per_year=24), mc_scheme="qe_m"
    )
    assert paths.scheme.startswith("heston_qe_m") and not paths.assets.flags.writeable
    chain = OptionChain(
        ttms=times[1:],
        forwards=np.array([100.0, 103.0]),
        discfactors=np.array([0.99, 0.97]),
        strikes_ttms=(np.array([100.0]), np.array([95.0, 108.0])),
        optiontypes_ttms=(np.array(["C"]), np.array(["P", "C"])),
        ids=None,
    )
    pricer = HestonPricer()
    kw = dict(mc_scheme="qe_m", nb_path=4096, steps_per_year=24, seed=19)
    means, errors = pricer.model_mc_price_chain(chain, p, **kw)
    for i, strikes in enumerate(chain.strikes_ttms):
        spots = (chain.forwards[i] * paths.assets[:, i + 1, 0])[:, None]
        payoff = chain.discfactors[i] * np.maximum(
            (spots - strikes) * np.where(chain.optiontypes_ttms[i] == "C", 1, -1), 0.0
        )
        np.testing.assert_allclose(means[i], payoff.mean(axis=0), rtol=1e-13)
        np.testing.assert_allclose(errors[i], payoff.std(axis=0, ddof=1) / 64, rtol=1e-13)
    x, v, qv = pricer.simulate_terminal_values(p, ttm=0.1, x0=0.3, **kw)
    for actual, key, offset in [
        (x, "log_zero_drift_return", 0.3),
        (v, "variance", 0.0),
        (qv, "quadratic_variance", 0.0),
    ]:
        np.testing.assert_array_equal(actual, paths.states[key][:, 1] + offset)
    vols = pricer.compute_mc_chain_implied_vols(chain, p, **kw)
    assert np.isfinite(vols[0][0]).all()
    with pytest.raises(ValueError, match="money-market"):
        pricer.model_mc_price_chain(chain, p, is_spot_measure=False, **kw)
    with pytest.raises(ValueError, match="log returns"):
        pricer.model_mc_price_chain(chain, p, variable_type=VariableType.Q_VAR, **kw)
    with pytest.raises(ValueError, match="mc_scheme"):
        pricer.simulate_terminal_values(p, mc_scheme="typo")


def test_absorbing_zero_and_zero_time():
    # A positive subnormal variance makes s²/m² exceed the float64 range.
    variance = np.array([1e-320])
    log_return = np.zeros(1)
    _qe_m_step(log_return, variance, np.zeros(1), 0., .5, -.9, 1., .1,
               np.zeros(1), np.zeros(1))
    assert variance[0] == 0. and np.isfinite(log_return[0])
    for ttm in (0.0, 0.25):
        x, v, q = HestonPricer().simulate_terminal_values(
            HestonParams(0.0, 0.0, 2.0, 1.0, 0.5),
            ttm=ttm,
            x0=0.2,
            nb_path=4,
            mc_scheme="qe_m",
            seed=7,
        )
        np.testing.assert_array_equal(x, np.full(4, 0.2))
        assert np.all(v == 0.0) and np.all(q == 0.0)


@pytest.mark.slow
def test_ten_year_wing_against_probability_inversion():
    estimate, variance, qvar = [], [], []
    for seed in range(917, 949):
        paths = HestonModel(HESTON_STRESS).simulate_paths(
            **_request(
                observation_times=np.array([0.0, 10.0]),
                spot0=100.0,
                n_paths=2048,
                steps_per_year=48,
                seed=seed,
            ),
            mc_scheme="qe_m",
        )
        spots = paths.assets[:, -1, 0]
        estimate.append([spots.mean(), np.maximum(spots - 140.0, 0.0).mean()])
        variance.append(paths.states["variance"][:, -1].mean())
        qvar.append(paths.states["quadratic_variance"][:, -1].mean())
    ref = _heston_probability_calls(10.0, HESTON_STRESS, np.array([140.0]))[0]
    obs = np.column_stack((estimate, variance, qvar))
    se = obs.std(axis=0, ddof=1) / np.sqrt(len(obs))
    assert se[1] <= 0.015
    assert np.all(
        abs(obs.mean(axis=0) - [100.0, ref, 0.04, 0.4]) <= 4 * se + [0.0, 0.012, 0.0, 0.0]
    )
