import numpy as np
from numba.typed import List
from typing import Tuple, Optional, Union


def de_pricer(ff, ff_transf) -> Tuple[np.ndarray, np.ndarray]:
    eps0 = 1e-6
    h = 0.5
    eps = 1e-6
    Nmax = 12.0
    maxlev = 7

    s = func(ff, 0.0)
    # add terms for k > 0
    n1, s = trunc_index(ff, h2=h, delta=1, s=s, Nmax=Nmax, eps0=eps0)
    # add terms for k < 0
    n2, s = trunc_index(ff, h2=-h, delta=1, s=s, Nmax=Nmax, eps0=eps0)
    # level 0 estimate
    model_prices_ttm_prev = h * s
    model_ivs_ttm_prev = ff_transf(model_prices_ttm_prev)[1]
    m = 0
    err_ivol = 1
    model_prices_ttm = None
    model_ivs_ttm = None
    for m in np.arange(1.0, maxlev):
        h = h / 2.0
        # add terms in refined mesh for k > 0
        s1 = part_sum(ff, h2=h, delta=2, N=n1)
        # add terms in refined mesh for k < 0
        s2 = part_sum(ff, h2=-h, delta=2, N=n2)
        model_prices_ttm = 0.5 * model_prices_ttm_prev + h * (s1 + s2)
        # print(f"model_prices: {model_prices_ttm[0]}")
        model_ivs_ttm = ff_transf(model_prices_ttm)[1]
        err_ivol = np.linalg.norm(model_ivs_ttm - model_ivs_ttm_prev)
        rel_diff = np.linalg.norm(model_prices_ttm - model_prices_ttm_prev) <= eps * np.linalg.norm(model_prices_ttm)

        if rel_diff or err_ivol <= 1e-6:
            break
        else:
            # update
            model_prices_ttm_prev = model_prices_ttm
            model_ivs_ttm_prev = model_ivs_ttm
            n1 = 2 * n1
            n2 = 2 * n2
    # we calculate prices of capped payoff => need to transform into call option prices
    model_prices_ttm = ff_transf(model_prices_ttm)[0]
    if m == maxlev - 1 and err_ivol > 1e-6:
        # raise ValueError(f"errvol = {err_ivol:.6f}, error is above tolerance")
        # print(f"errvol = {err_ivol:.6f}, error is above tolerance")
        pass
    return model_prices_ttm, model_ivs_ttm


def func(ff, x: Union[float, np.ndarray]) -> np.ndarray:
    """calculate the term w_k * f(x_k) of the DE scheme
    x must equal kh for some k"""
    if isinstance(x, float):
        x = np.array([x])
    half_pi = 0.5 * np.pi
    exp_x = np.exp(x)
    sinh_x = 0.5 * (exp_x - 1.0 / exp_x)
    cosh_x = 0.5 * (exp_x + 1.0 / exp_x)
    exp_sinh_x = np.exp(half_pi * sinh_x)
    w_k = half_pi * cosh_x * exp_sinh_x
    x_k = exp_sinh_x
    val = (ff(x_k).T * w_k).T
    return val


def part_sum(ff, h2: float, delta: int, N: int) -> float:
    s = 0.0
    func_vals = func(ff, h2 + np.arange(0.0, N, 1.0) * delta * h2)
    for idx, func_val in enumerate(func_vals):
        s = s + func_vals[idx]
    return s

def trunc_index(ff,
                h2: float,
                delta: int,
                s: np.ndarray,
                Nmax: float,
                eps0: float) -> (int, np.ndarray):
    x = h2
    k = 1
    for k in np.arange(1.0, Nmax):
        xi = func(ff, x)
        s = s + xi
        if np.all(np.linalg.norm(xi, axis=0) <= eps0 * np.linalg.norm(s, axis=0)):
            break
        x = x + delta * h2
    return k, s


