"""Offline European-call reference cases for Fourier integration.

Premiums use forward 100 and unit discount. Heston's independent P1/P2 inversion
tests price accuracy. LogSV holds the second-order transform fixed across grids
to isolate quadrature error from expansion error.
"""

import numpy as np
import pytest
from scipy.integrate import quad_vec, simpson
from scipy.special import roots_legendre
import vanilla_option_pricers as vop

from stochvolmodels.models.heston import HestonModel
from stochvolmodels.models.logsv.affine_expansion import ExpansionOrder, compute_logsv_a_mgf_grid
from stochvolmodels.models.logsv.params import LogSvParams
from stochvolmodels.pricers.heston_pricer import HestonParams, HestonPricer


FORWARD = 100.0
PREMIUM_ATOL = 1e-6
HESTON_STRIKES = np.array([60.0, 70.0, 100.0, 140.0])
HESTON_STRESS = HestonParams(v0=0.04, theta=0.04, kappa=0.5, rho=-0.9, volvol=1.0)
# PyFENG HestonFft benchmark, ten-year European calls at zero carry.
# https://github.com/quants-net/PyFENG/blob/main/pyfeng/sv_fft.py
HESTON_PUBLISHED = np.array([44.32997507, 35.84976970, 13.08467014, 0.29577444])

LOGSV_STRIKES = np.array([80.0, 100.0, 120.0])
LOGSV_BASE = LogSvParams(sigma0=0.40, theta=0.34, kappa1=2.0, kappa2=3.0, beta=-0.40, volvol=0.65)
LOGSV_ALT = LogSvParams(sigma0=0.28, theta=0.40, kappa1=2.0, kappa2=3.0, beta=-0.65, volvol=0.85)
PANELS_TO_128 = np.array([0, 0.5, 1, 2, 4, 8, 16, 32, 64, 128])
PANELS_TO_512 = np.r_[PANELS_TO_128[:8], np.arange(64, 513, 32)]


def _heston_cf(z: complex, ttm: float, params: HestonParams) -> complex:
    """Characteristic function of log(S_T/F) in the P1/P2 convention."""
    eta = params.volvol
    drift = params.kappa - 1j * params.rho * eta * z
    root = np.sqrt(drift**2 + eta**2 * (z**2 + 1j * z))
    ratio = (drift - root) / (drift + root)
    decay = np.exp(-root * ttm)
    intercept = params.kappa * params.theta / eta**2 * (
        (drift - root) * ttm - 2.0 * np.log((1.0 - ratio * decay) / (1.0 - ratio))
    )
    loading = (drift - root) / eta**2 * (1.0 - decay) / (1.0 - ratio * decay)
    return np.exp(intercept + params.v0 * loading)


def _heston_probability_calls(ttm: float, params: HestonParams, strikes: np.ndarray) -> np.ndarray:
    """Invert F*P1 - K*P2 on [0, infinity), independently of SVM's transform and grid."""
    log_moneyness = np.log(FORWARD / strikes)

    def integrand(u: float) -> np.ndarray:
        # The normalized forward has unit mean, hence phi(-i) = 1 in P1.
        tilted = FORWARD * _heston_cf(u - 1j, ttm, params)
        ordinary = strikes * _heston_cf(u, ttm, params)
        return np.imag(np.exp(1j * u * log_moneyness) * (tilted - ordinary)) / (np.pi * u)

    correction, error = quad_vec(integrand, 0.0, np.inf, epsabs=1e-11, epsrel=1e-12)
    assert error < 1e-9
    return 0.5 * (FORWARD - strikes) + correction


def _lewis_integrand(u: np.ndarray, log_mgf: np.ndarray, strikes: np.ndarray) -> np.ndarray:
    """Return the integrand subtracted from the forward on SVM's negative half contour."""
    phase = np.exp(-1j * np.log(FORWARD / strikes)[:, None] * u)
    kernel = np.exp(log_mgf) / (np.pi * (0.25 + u**2))
    return np.sqrt(FORWARD * strikes)[:, None] * np.real(phase * kernel)


def _heston_simpson_calls(
    ttm: float, params: HestonParams, cutoff: float, nodes: int,
) -> np.ndarray:
    """Use the package transform and composite Simpson inversion at an explicit resolution."""
    u = np.linspace(0.0, cutoff, nodes)
    log_mgf = HestonModel(params).log_mgf_grid(ttm=ttm, phi_grid=-0.5 + 1j * u)
    return FORWARD - simpson(_lewis_integrand(u, log_mgf, HESTON_STRIKES), x=u, axis=-1)


def test_heston_published_prices_and_legacy_wing() -> None:
    """The historical 1,000-node price misses the published four-strike benchmark in the wing."""
    reference = _heston_probability_calls(10.0, HESTON_STRESS, HESTON_STRIKES)
    np.testing.assert_allclose(reference, HESTON_PUBLISHED, atol=5e-9, rtol=0)
    legacy, _ = HestonPricer().price_slice(
        params=HESTON_STRESS, ttm=10.0, forward=FORWARD, strikes=HESTON_STRIKES,
        optiontypes=np.full(4, "C"),
    )
    np.testing.assert_allclose(
        legacy, [44.32625418, 35.85852936, 13.06885163, 0.22052944], atol=1e-8, rtol=0,
    )
    assert reference[-1] - legacy[-1] > 0.075


def test_heston_mesh_refinement_requires_wider_cutoff() -> None:
    """Resolving the same truncated interval leaves a 0.075-premium wing error."""
    cutoff = 5.6 / 0.3
    reference = _heston_probability_calls(10.0, HESTON_STRESS, HESTON_STRIKES)
    coarse = _heston_simpson_calls(10.0, HESTON_STRESS, cutoff, 1_001)
    dense = _heston_simpson_calls(10.0, HESTON_STRESS, cutoff, 4_001)
    np.testing.assert_allclose(dense, coarse, atol=1e-9, rtol=0)
    assert reference[-1] - dense[-1] > 0.075

    wide = _heston_simpson_calls(10.0, HESTON_STRESS, cutoff=160.0, nodes=3_201)
    np.testing.assert_allclose(wide, reference, atol=PREMIUM_ATOL, rtol=0)


@pytest.mark.parametrize(
    "ttm,kappa,rho,volvol",
    [(1.0, 2.0, -0.7, 0.3), (7 / 365, 2.0, -0.7, 0.3),
     (10.0, 0.5, -0.97, 1.0), (3.0, 1.0, 0.7, 0.6)],
    ids=["one-year", "one-week", "strong-negative-correlation", "positive-correlation"],
)
def test_heston_control_prices(ttm: float, kappa: float, rho: float, volvol: float) -> None:
    """Check the probability inversion against a separately resolved Lewis inversion."""
    params = HestonParams(v0=0.04, theta=0.04, kappa=kappa, rho=rho, volvol=volvol)
    reference = _heston_probability_calls(ttm, params, HESTON_STRIKES)
    wide = _heston_simpson_calls(ttm, params, cutoff=1024.0, nodes=20_481)
    np.testing.assert_allclose(wide, reference, atol=PREMIUM_ATOL, rtol=0)


# Positive half and origin of QUADPACK DQK15's fixed Kronrod rule on [-1, 1].
# These tabulated test inputs reproduce the 135- and 330-node experiments.
_KRONROD_NONNEGATIVE = np.array([
    [0.0,                0.2094821410847278],
    [0.2077849550078985, 0.2044329400752989],
    [0.4058451513773972, 0.1903505780647854],
    [0.5860872354676911, 0.1690047266392679],
    [0.7415311855993944, 0.1406532597155259],
    [0.8648644233597691, 0.1047900103222502],
    [0.9491079123427585, 0.06309209262997855],
    [0.9914553711208126, 0.02293532201052922],
])
_x, _w = _KRONROD_NONNEGATIVE.T
KRONROD_15 = (np.r_[-_x[:0:-1], _x], np.r_[_w[:0:-1], _w])


def _panel_grid(
    edges: np.ndarray, rule: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Map a fixed reference rule to each panel, keeping its independently supplied weights."""
    nodes, weights = rule
    midpoints = 0.5 * (edges[1:] + edges[:-1])
    half_widths = 0.5 * np.diff(edges)
    u = midpoints[:, None] + half_widths[:, None] * nodes
    return u.ravel(), (half_widths[:, None] * weights).ravel()


def _legendre_reference_grid(cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    """Bisect the dyadic panels through the wider cutoff and apply 24-point Gauss–Legendre."""
    edges = np.r_[0.0, 2.0 ** np.arange(-1, int(np.log2(cutoff)) + 1)]
    mids = 0.5 * (edges[:-1] + edges[1:])
    return _panel_grid(np.sort(np.r_[edges, mids]), roots_legendre(24))


def _logsv_calls(
    params: LogSvParams, expiry_days: tuple[int, ...], grid: tuple[np.ndarray, np.ndarray],
    strikes: np.ndarray = LOGSV_STRIKES,
) -> np.ndarray:
    """Price successive expiries with one second-order expansion per grid and a flat backbone."""
    u, weights = grid
    phi = -0.5 + 1j * u
    zero = np.zeros_like(phi)
    rows = []
    state = None
    previous_day = 0
    for day in expiry_days:
        state, log_mgf = compute_logsv_a_mgf_grid(
            ttm=(day - previous_day) / 365.0, phi_grid=phi, psi_grid=zero, theta_grid=zero,
            expansion_order=ExpansionOrder.SECOND, a_t0=state, **params.to_dict(),
        )
        rows.append(FORWARD - _lewis_integrand(u, log_mgf, strikes) @ weights)
        previous_day = day
    return np.array(rows)


@pytest.mark.slow
@pytest.mark.parametrize(
    "params,expected",
    [
        (LOGSV_BASE, [[20.16800278, 4.48351895, 0.18580654],
                      [22.96248151, 10.24023560, 3.47857423]]),
        (LOGSV_ALT, [[20.05489509, 3.34717612, 0.01493427],
                     [22.63551238, 9.25630717, 2.45199551]]),
    ],
    ids=["base", "alternate"],
)
def test_logsv_monthly_grid_against_legendre(
    params: LogSvParams, expected: list[list[float]],
) -> None:
    """The 135-node frozen grid resolves both 30- and 180-day calls for these parameters."""
    prices = _logsv_calls(params, (30, 180), _panel_grid(PANELS_TO_128, KRONROD_15))
    reference = _logsv_calls(params, (30, 180), _legendre_reference_grid(256))
    # Checked separately with the IJTAF paper's Black control-variate Fourier inversion.
    np.testing.assert_allclose(reference, expected, atol=PREMIUM_ATOL, rtol=0)
    np.testing.assert_allclose(prices, reference, atol=PREMIUM_ATOL, rtol=0)


@pytest.mark.slow
def test_logsv_one_day_tail_requires_wider_grid() -> None:
    """The one-day ATM premium reveals a missing tail on the otherwise accurate monthly grid."""
    days = (1, 7)
    reference = _logsv_calls(LOGSV_ALT, days, _legendre_reference_grid(1024))
    narrow = _logsv_calls(LOGSV_ALT, days, _panel_grid(PANELS_TO_128, KRONROD_15))
    assert reference[0, 1] == pytest.approx(0.58564723, abs=PREMIUM_ATOL, rel=0)
    assert narrow[0, 1] - reference[0, 1] == pytest.approx(0.00729765, abs=PREMIUM_ATOL, rel=0)

    refined = _logsv_calls(LOGSV_ALT, days, _panel_grid(PANELS_TO_512, KRONROD_15))
    np.testing.assert_allclose(refined, reference, atol=PREMIUM_ATOL, rtol=0)


def test_logsv_constant_volatility_against_black() -> None:
    """At equilibrium with both volatility diffusion loadings zero, LogSV is Black exactly."""
    vol = 0.3
    params = LogSvParams(sigma0=vol, theta=vol, kappa1=2.0, kappa2=3.0, beta=0.0, volvol=0.0)
    prices = _logsv_calls(params, (30,), _legendre_reference_grid(256))[0]
    expected = vop.compute_bsm_vanilla_slice_prices(
        30 / 365, FORWARD, LOGSV_STRIKES, np.full(3, vol), np.full(3, "C"), 1.0,
    )
    np.testing.assert_allclose(prices, expected, atol=PREMIUM_ATOL, rtol=0)
