"""
Verification script for the review chapter (model-choice chapter) analysis of 2026-08-25.

Model under P with beta = 0:
    d sigma = (kappa1 + kappa2 sigma)(theta - sigma) dt + vartheta sigma dW,
    y = ln sigma: dy = [kappa1 theta e^{-y} + (kappa2 theta - kappa1) - kappa2 e^{y} - vartheta^2/2] dt + vartheta dW.

Claims checked:
    C1  the stationary density of y from the discretised generator equals the GIG-type density
        g(y) = c exp(eta y - q e^{-y} - b e^{y}) with q = 2 kappa1 theta / vartheta^2,
        b = 2 kappa2 / vartheta^2, eta = 2 (kappa2 theta - kappa1) / vartheta^2 - 1;
    C2  the ACF of sigma from the spectral decomposition of the generator equals the Monte Carlo ACF;
    C3  time scaling: (c kappa1, c kappa2, c vartheta^2) leaves the density unchanged and maps
        ACF(h) to ACF(c h), so the density identifies parameters only up to a clock and the ACF sets the clock;
    C4  the weights of the leading eigenmodes, which measure how far the one-factor ACF is from
        a single exponential, and the leading rate lambda_1 against kappa_lin = kappa1 + kappa2 theta;
    C5  a power-law fit 1 - c h^{2 alpha + 1} on daily lags 0..120 to the one-factor MODEL ACF
        returns a negative alpha, so the daily-lag alpha does not discriminate rough from Markov.

Units: time in years, one trading day = 1/260 year, volatility in decimals.
Dependencies: numpy and scipy only. The 2023 fitted parameters are those hard-coded in
../ssrn_2023/article_figures.py (untraced vintage, see the analysis note).
Not in this module: any data loading, plotting, or model calibration.
"""
# packages
import numpy as np
from scipy import special as sps
from scipy.linalg import eigh
from scipy.optimize import minimize
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

DAYS_PER_YEAR = 260.0


@dataclass(frozen=True)
class LogSvStationaryParams:
    theta: float      # mean volatility level, decimals
    kappa1: float     # linear mean-reversion, 1/years
    kappa2: float     # quadratic mean-reversion, 1/(years x vol)
    vartheta: float   # total vol-of-vol, sqrt(beta^2 + eps^2)


PARAMS_2023: Dict[str, LogSvStationaryParams] = {
    'VIX': LogSvStationaryParams(theta=0.19928505844247962, kappa1=1.2878835150774184,
                                 kappa2=1.9267876555824357, vartheta=0.7210463316739526),
    'MOVE': LogSvStationaryParams(theta=0.9109917133860931, kappa1=0.1,
                                  kappa2=0.41131244621275886, vartheta=0.3564212939473691),
    'OVX': LogSvStationaryParams(theta=0.3852514800317871, kappa1=2.7774564907918027,
                                 kappa2=2.2351296851221107, vartheta=0.8344408577025486),
    'BTC': LogSvStationaryParams(theta=0.7118361434192538, kappa1=2.214702576955766,
                                 kappa2=2.18028273418397, vartheta=0.921487415907961),
}


def gig_shape(params: LogSvStationaryParams) -> Tuple[float, float, float]:
    """Return (q, b, eta): the scale-free shape parameters of the stationary density."""
    v2 = params.vartheta ** 2
    q = 2.0 * params.kappa1 * params.theta / v2
    b = 2.0 * params.kappa2 / v2
    eta = 2.0 * (params.kappa2 * params.theta - params.kappa1) / v2 - 1.0
    return q, b, eta


def theta_from_shape(q: float, b: float, eta: float) -> float:
    """theta solves b theta^2 - (eta + 1) theta - q = 0, so theta is scale-free."""
    return ((eta + 1.0) + np.sqrt((eta + 1.0) ** 2 + 4.0 * b * q)) / (2.0 * b)


def stationary_log_density(y: np.ndarray, params: LogSvStationaryParams) -> np.ndarray:
    """Normalised stationary density of y = ln sigma."""
    q, b, eta = gig_shape(params=params)
    if q <= 0.0 or b <= 0.0:
        raise ValueError(f"q={q}, b={b} must be positive (kappa1 > 0, kappa2 > 0)")
    c = np.power(b / q, eta / 2.0) / (2.0 * sps.kv(eta, 2.0 * np.sqrt(q * b)))
    return c * np.exp(eta * y - (q * np.exp(-y) + b * np.exp(y)))


def unnormalised_log_of_density(y: np.ndarray, params: LogSvStationaryParams) -> np.ndarray:
    q, b, eta = gig_shape(params=params)
    return eta * y - (q * np.exp(-y) + b * np.exp(y))


def spectral_acf(params: LogSvStationaryParams,
                 lags_years: np.ndarray,
                 y_grid: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ACF of sigma = e^y from the generator L f = (1/pi) d/dy[(vartheta^2/2) pi f'].

    The detailed-balance discretisation with midpoint densities makes S = D^{1/2} L D^{-1/2}
    an exactly symmetric tridiagonal matrix. Returns (acf, rates, weights) with
    acf(h) = sum_n weights_n exp(-rates_n h) and sum_n weights_n = 1.
    """
    if np.any(np.diff(lags_years) < 0.0):
        raise ValueError(f"lags must be non-decreasing: {lags_years}")
    n = len(y_grid)
    dy = y_grid[1] - y_grid[0]
    lp = unnormalised_log_of_density(y=y_grid, params=params)
    y_mid = 0.5 * (y_grid[1:] + y_grid[:-1])
    lp_mid = unnormalised_log_of_density(y=y_mid, params=params)
    d = 0.5 * params.vartheta ** 2
    off = d / dy ** 2 * np.exp(lp_mid - 0.5 * (lp[1:] + lp[:-1]))
    diag = np.zeros(n)
    diag[:-1] -= d / dy ** 2 * np.exp(lp_mid - lp[:-1])
    diag[1:] -= d / dy ** 2 * np.exp(lp_mid - lp[1:])
    s_matrix = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    pi = np.exp(lp - lp.max())
    pi = pi / pi.sum()
    rates, modes = eigh(-s_matrix)
    sigma = np.exp(y_grid)
    coef = modes.T @ (np.sqrt(pi) * sigma)
    mean = float(pi @ sigma)
    var = float(pi @ sigma ** 2 - mean ** 2)
    mask = rates > 1e-8
    rates = rates[mask]
    weights = coef[mask] ** 2 / var
    acf = np.array([np.sum(weights * np.exp(-rates * h)) for h in lags_years])
    return acf, rates, weights


def mc_acf(params: LogSvStationaryParams,
           nb_lags: int = 121,
           nb_years: float = 400.0,
           dt: float = 1.0 / DAYS_PER_YEAR,
           seed: int = 3
           ) -> np.ndarray:
    """Sample ACF of sigma from one long Euler path in log coordinates."""
    rng = np.random.default_rng(seed)
    n = int(nb_years / dt)
    y = np.log(params.theta)
    ys = np.empty(n)
    v = params.vartheta
    sq_dt = np.sqrt(dt)
    for i in range(n):
        drift = (params.kappa1 * params.theta * np.exp(-y) + (params.kappa2 * params.theta - params.kappa1)
                 - params.kappa2 * np.exp(y) - 0.5 * v ** 2)
        y = y + drift * dt + v * sq_dt * rng.standard_normal()
        ys[i] = y
    s = np.exp(ys)
    s = s - s.mean()
    var = np.mean(s ** 2)
    return np.array([np.mean(s[h:] * s[:len(s) - h]) / var for h in range(nb_lags)])


def fit_power_law(acf: np.ndarray, lags_years: np.ndarray) -> Tuple[float, float]:
    """Fit ACF(h) = 1 - c h^{2 alpha + 1} by least squares, the estimator used in the 2023 draft."""
    def objective(x: np.ndarray) -> float:
        alpha, c = x
        return float(np.nansum((1.0 - c * np.power(lags_years, 2.0 * alpha + 1.0) - acf) ** 2))
    res = minimize(objective, x0=np.array([0.1, 0.99]), bounds=((-0.5, 0.5), (0.01, 1.5)), method='L-BFGS-B')
    return float(res.x[0]), float(res.x[1])


class LocalTests(Enum):
    C1_STATIONARY_DENSITY = 1
    C2_SPECTRAL_VS_MC = 2
    C3_TIME_SCALING = 3
    C4_EIGEN_WEIGHTS = 4
    C5_POWER_LAW_ON_MODEL_ACF = 5


def run_local_test(local_test: LocalTests) -> None:
    y_grid = np.linspace(np.log(0.02), np.log(6.0), 800)
    lags = np.arange(0, 121) / DAYS_PER_YEAR

    if local_test == LocalTests.C1_STATIONARY_DENSITY:
        for name, params in PARAMS_2023.items():
            q, b, eta = gig_shape(params=params)
            g = stationary_log_density(y=y_grid, params=params)
            mass = np.trapezoid(g, y_grid) if hasattr(np, 'trapezoid') else np.trapz(g, y_grid)
            print(f"{name}: q={q:.3f}, b={b:.3f}, eta={eta:.3f}, mass={mass:.5f}, "
                  f"theta from (q,b,eta)={theta_from_shape(q, b, eta):.4f} vs theta={params.theta:.4f}")

    elif local_test == LocalTests.C2_SPECTRAL_VS_MC:
        params = PARAMS_2023['VIX']
        acf_s, _, _ = spectral_acf(params=params, lags_years=lags, y_grid=y_grid)
        acf_m = mc_acf(params=params, nb_lags=121, nb_years=400.0)
        for h in [1, 5, 20, 60, 120]:
            print(f"lag {h:3d} days: spectral {acf_s[h]:.4f}, MC {acf_m[h]:.4f}")

    elif local_test == LocalTests.C3_TIME_SCALING:
        params = PARAMS_2023['VIX']
        scaled = LogSvStationaryParams(theta=params.theta, kappa1=2.0 * params.kappa1,
                                       kappa2=2.0 * params.kappa2, vartheta=np.sqrt(2.0) * params.vartheta)
        acf_scaled, _, _ = spectral_acf(params=scaled, lags_years=lags, y_grid=y_grid)
        acf_2h, _, _ = spectral_acf(params=params, lags_years=2.0 * lags, y_grid=y_grid)
        g1 = stationary_log_density(y=y_grid, params=params)
        g2 = stationary_log_density(y=y_grid, params=scaled)
        print(f"max|ACF_scaled(h) - ACF(2h)| = {np.max(np.abs(acf_scaled - acf_2h)):.2e}, "
              f"max|density difference| = {np.max(np.abs(g1 - g2)):.2e}")

    elif local_test == LocalTests.C4_EIGEN_WEIGHTS:
        for name, params in PARAMS_2023.items():
            acf_s, rates, weights = spectral_acf(params=params, lags_years=lags, y_grid=y_grid)
            idx = np.argsort(rates)[:3]
            kappa_lin = params.kappa1 + params.kappa2 * params.theta
            half_life_days = DAYS_PER_YEAR * np.log(2.0) / rates[idx[0]]
            print(f"{name}: kappa_lin={kappa_lin:.3f}, rates={np.round(rates[idx], 3)}, "
                  f"weights={np.round(weights[idx], 3)}, half-life={half_life_days:.0f} days, "
                  f"ACF(1d)={acf_s[1]:.4f}, ACF(120d)={acf_s[120]:.3f}")

    elif local_test == LocalTests.C5_POWER_LAW_ON_MODEL_ACF:
        for name, params in PARAMS_2023.items():
            acf_s, _, _ = spectral_acf(params=params, lags_years=lags, y_grid=y_grid)
            alpha, c = fit_power_law(acf=acf_s, lags_years=lags)
            resid = np.max(np.abs(1.0 - c * np.power(lags, 2.0 * alpha + 1.0) - acf_s))
            print(f"{name}: alpha={alpha:.3f}, c={c:.3f}, max abs residual={resid:.3f}")


if __name__ == '__main__':
    for local_test in LocalTests:
        print(f"\n{local_test.name}")
        run_local_test(local_test=local_test)
