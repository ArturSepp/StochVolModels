"""Canonical script of docs/app_impermanent_loss_hedging.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here. The impermanent loss of a concentrated-liquidity position is replicated by
European payoffs and valued with the log-normal SV transform, following the module
``papers/il_hedging/run_logsv_for_il_payoff.py`` that accompanies Lipton, Lucic and Sepp (2025).
Select a case in ``Locals`` or run the file to execute every case.
"""
from enum import Enum

import numpy as np
from numba import njit

import stochvolmodels as svm
from stochvolmodels.utils.funcs import set_seed

# parameters of the module's example: a range of 2,000 to 2,400 around a price of 2,200, ten days
PARAMS = svm.LogSvParams(sigma0=0.4862, theta=0.6176, kappa1=1.9558, kappa2=1.9784, beta=-0.2692,
                         volvol=3.2658)
P0, PA, PB, TTM = 2200.0, 2000.0, 2400.0, 10.0 / 365.0


def impermanent_loss(p: np.ndarray, p0: float = P0, pa: float = PA, pb: float = PB) -> np.ndarray:
    """Position value minus the value of holding the initial tokens, per unit of liquidity."""
    position = np.where(p < pa, p * (1.0 / np.sqrt(pa) - 1.0 / np.sqrt(pb)),
                        np.where(p > pb, np.sqrt(pb) - np.sqrt(pa),
                                 2.0 * np.sqrt(p) - np.sqrt(pa) - p / np.sqrt(pb)))
    hold = p * (1.0 / np.sqrt(p0) - 1.0 / np.sqrt(pb)) + np.sqrt(p0) - np.sqrt(pa)
    return position - hold


def replication(p: np.ndarray, p0: float = P0, pa: float = PA, pb: float = PB) -> dict:
    """The static portfolio whose payoff is minus the impermanent loss, by component."""
    return {"square root in range": -2.0 * np.sqrt(p) * ((p >= pa) & (p <= pb)),
            "linear": p / np.sqrt(p0) + np.sqrt(p0),
            "put at pa": np.maximum(pa - p, 0.0) / np.sqrt(pa),
            "call at pb": -np.maximum(p - pb, 0.0) / np.sqrt(pb),
            "digital put at pa": -2.0 * np.sqrt(pa) * (p < pa),
            "digital call at pb": -2.0 * np.sqrt(pb) * (p > pb)}


@njit(cache=False)
def square_root_in_range(log_mgf: np.ndarray, phi_grid: np.ndarray, weights: np.ndarray,
                         forward: float, pa: float, pb: float) -> float:
    """E[sqrt(P_T) 1(pa <= P_T <= pb)] by Fourier inversion, as the module computes it."""
    x, xa, xb = np.log(forward), np.log(pa), np.log(pb)
    transform = (np.exp((phi_grid + 0.5) * xb - phi_grid * x)
                 - np.exp((phi_grid + 0.5) * xa - phi_grid * x)) / (phi_grid + 0.5)
    return np.nansum(np.real((weights / np.pi) * transform * np.exp(log_mgf)))


def expected_loss(params: svm.LogSvParams = PARAMS, forward: float = P0, p0: float = P0,
                  pa: float = PA, pb: float = PB, ttm: float = TTM) -> float:
    """Expected impermanent loss per unit of initial position value, from the transform."""
    phi_grid, psi_grid, theta_grid = svm.get_transform_var_grid(
        vol_scaler=params.sigma0 * np.sqrt(min(ttm, 0.5 / 12.0)), real_phi=-0.4, max_phi=1001)
    _, log_mgf = svm.compute_logsv_a_mgf_grid(ttm=ttm, phi_grid=phi_grid, psi_grid=psi_grid,
                                              theta_grid=theta_grid, **params.to_dict())
    put, call = svm.vanilla_slice_pricer_with_mgf_grid(
        log_mgf_grid=log_mgf, phi_grid=phi_grid, forward=forward, strikes=np.array([pa, pb]),
        optiontypes=np.array(["P", "C"]))
    digital_put, digital_call = svm.digital_slice_pricer_with_mgf_grid(
        log_mgf_grid=log_mgf, phi_grid=phi_grid, forward=forward, strikes=np.array([pa, pb]),
        optiontypes=np.array(["P", "C"]))
    root = square_root_in_range(log_mgf, phi_grid, svm.compute_integration_weights(phi_grid),
                                forward, pa, pb)
    minus_loss = (-2.0 * root + forward / np.sqrt(p0) + np.sqrt(p0) + put / np.sqrt(pa)
                  - call / np.sqrt(pb) - 2.0 * np.sqrt(pa) * digital_put
                  - 2.0 * np.sqrt(pb) * digital_call)
    position0 = 2.0 * np.sqrt(p0) - p0 / np.sqrt(pb) - np.sqrt(pa)
    return -minus_loss / position0


def monte_carlo_loss(params: svm.LogSvParams = PARAMS, nb_path: int = 400000,
                     seed: int = 17) -> tuple:
    """Monte Carlo expected loss per unit of initial position value, with its standard error."""
    set_seed(seed)
    x, _, _ = svm.LogSVPricer().simulate_terminal_values(params=params, ttm=TTM, nb_path=nb_path)
    position0 = 2.0 * np.sqrt(P0) - P0 / np.sqrt(PB) - np.sqrt(PA)
    loss = impermanent_loss(P0 * np.exp(x)) / position0
    return np.mean(loss), np.std(loss) / np.sqrt(nb_path)


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    REPLICATION = 1
    VALUE = 2
    SENSITIVITY = 3


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    if local == Locals.REPLICATION:
        prices = np.linspace(1500.0, 3000.0, 3001)
        loss = impermanent_loss(prices)
        # the portfolio pays minus the loss at every terminal price
        assert np.max(np.abs(loss + sum(replication(prices).values()))) < 1e-12
        assert abs(impermanent_loss(np.array([P0]))[0]) < 1e-12 and np.all(loss <= 1e-12)

    elif local == Locals.VALUE:
        value = expected_loss()
        mc, error = monte_carlo_loss()
        print(value, mc, error)
        np.testing.assert_allclose(value, -0.017397, atol=1e-6)
        assert abs(value - mc) < 3.0 * error

    elif local == Locals.SENSITIVITY:
        vartheta2 = PARAMS.beta ** 2 + PARAMS.volvol ** 2
        by_beta = [expected_loss(svm.LogSvParams(
            sigma0=PARAMS.sigma0, theta=PARAMS.theta, kappa1=PARAMS.kappa1, kappa2=PARAMS.kappa2,
            beta=beta, volvol=np.sqrt(vartheta2 - beta ** 2))) for beta in (-1.0, 0.0, 1.0)]
        by_volvol = [expected_loss(svm.LogSvParams(
            sigma0=PARAMS.sigma0, theta=PARAMS.theta, kappa1=PARAMS.kappa1, kappa2=PARAMS.kappa2,
            beta=PARAMS.beta, volvol=volvol)) for volvol in (1.0, 4.0)]
        print(by_beta, by_volvol)
        # at a fixed total vol-of-vol the volatility beta moves the loss little; vol-of-vol more
        np.testing.assert_allclose(by_beta, [-0.01726, -0.01742, -0.01734], atol=1e-5)
        np.testing.assert_allclose(by_volvol, [-0.01650, -0.01786], atol=1e-5)


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
