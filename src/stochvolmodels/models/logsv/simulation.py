"""LogSV terminal-state propagation and reproducible chain random tapes.

The existing Euler volatility update, supplied-normal convention and quadratic
variance accumulation are preserved. Chain tapes use a local RandomState and
retain the observation partition and W0-before-W1 draw ordering.
"""

from typing import Tuple, Optional
import numpy as np
from numba import njit
from numba.typed import List
from stochvolmodels.utils.funcs import set_time_grid


@njit(cache=False, fastmath=False)
def simulate_logsv_x_vol_terminal(
    ttm: float,
    x0: np.ndarray,
    sigma0: np.ndarray,
    qvar0: np.ndarray,
    theta: float,
    kappa1: float,
    kappa2: float,
    beta: float,
    volvol: float,
    vol_backbone_eta: float = 1.0,
    is_spot_measure: bool = True,
    nb_path: int = 100000,
    nb_steps_per_year: int = 360,
    W0: Optional[np.ndarray] = None,
    W1: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    simulate terminal log-return, volatility and quadratic variance.

    Implements the discretization of Corollary 3.5, Eq. (3.59): the log-price
    X_t of Eq. (3.16), the log-volatility L_t of Eq. (3.55) and the quadratic
    variance I_t of Eq. (3.12), driven by two independent Brownian motions
    W^(0) and W^(1). Under the inverse measure the sign of the X drift flips, per
    the dynamics of Eq. (3.36).

    Parameters
    ----------
    ttm : float
        Horizon in years.
    x0, sigma0, qvar0 : np.ndarray
        Initial values, either scalars broadcast to nb_path or full path vectors,
        so that a chain can be simulated maturity by maturity from the previous
        terminal state.
    theta, kappa1, kappa2, beta, volvol : float
        Model parameters of Eq. (3.12).
    vol_backbone_eta : float, default 1.0
        Deterministic loading on price diffusion; theta is unchanged. 1.0 is the core model.
    is_spot_measure : bool, default True
        MMA measure when True, inverse measure when False.
    nb_path : int, default 100000
        Number of paths. Figs. 6 and 9 use 400,000.
    nb_steps_per_year : int, default 360
        Steps per year.
    W0, W1 : Optional[np.ndarray], default None
        Pre-drawn scaled Brownian increments. Supplying them fixes the randoms
        across calibration iterations.
    dt : Optional[float], default None
        Step size matching W0 and W1 when those are supplied.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Terminal log-return, volatility and quadratic variance, each of length
        nb_path.
    """
    if x0.shape[0] == 1:  # initial value
        x0 = np.full(nb_path, x0[0])
    else:
        assert x0.shape[0] == nb_path

    if qvar0.shape[0] == 1:  # broadcast the supplied initial or continuing state
        qvar0 = np.full(nb_path, qvar0[0])
    else:
        assert qvar0.shape[0] == nb_path

    if sigma0.shape[0] == 1:
        sigma0 = sigma0 * np.ones(nb_path)
    else:
        assert sigma0.shape[0] == nb_path
    if W0 is None and W1 is None:
        nb_steps1, dt, grid_t = set_time_grid(ttm=ttm, nb_steps_per_year=nb_steps_per_year)
        # print(f"nb_steps1={nb_steps1}, dt={dt}")
        sdt = np.sqrt(dt)
        W0_ = sdt * np.random.normal(0, 1, size=(nb_steps1, nb_path))
        W1_ = sdt * np.random.normal(0, 1, size=(nb_steps1, nb_path))
    else:
        sdt = np.sqrt(dt)
        W0_ = sdt * W0
        W1_ = sdt * W1

    if is_spot_measure:
        alpha, adj = -1.0, 0.0
    else:
        alpha, adj = 1.0, beta * vol_backbone_eta  # ? vol_backbone_eta

    vartheta2 = beta * beta + volvol * volvol
    vol_backbone_eta2 = vol_backbone_eta * vol_backbone_eta
    vol_var = np.log(sigma0)
    for t_, (w0, w1) in enumerate(zip(W0_, W1_)):
        sigma0_2dt = vol_backbone_eta2 * sigma0 * sigma0 * dt
        x0 = x0 + alpha * 0.5 * sigma0_2dt + vol_backbone_eta * sigma0 * w0
        vol_var = (
            vol_var
            + (
                (kappa1 * theta / sigma0 - kappa1)
                + kappa2 * (theta - sigma0)
                + adj * sigma0
                - 0.5 * vartheta2
            )
            * dt
            + beta * w0
            + volvol * w1
        )
        sigma0 = np.exp(vol_var)
        qvar0 = qvar0 + 0.5 * (sigma0_2dt + vol_backbone_eta2 * sigma0 * sigma0 * dt)

    return x0, sigma0, qvar0


def get_randoms_for_chain_valuation(
    ttms: np.ndarray, nb_path: int = 100000, nb_steps_per_year: int = 360, seed: int = 10
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Draw fixed random normals for Monte Carlo chain valuation.

    A local legacy NumPy generator owns the seed, so repeated calls replay the
    same arrays without mutating NumPy's process-global random state.
    """
    rng = np.random.RandomState(seed)
    W0s = List()
    W1s = List()
    dts = List()
    ttm0 = 0.0
    for ttm in ttms:
        # qqq
        nb_steps_, dt, grid_t = set_time_grid(ttm=ttm - ttm0, nb_steps_per_year=nb_steps_per_year)
        W0s.append(rng.normal(0, 1, size=(nb_steps_, nb_path)))
        W1s.append(rng.normal(0, 1, size=(nb_steps_, nb_path)))
        dts.append(dt)
        ttm0 = ttm
    return W0s, W1s, dts
