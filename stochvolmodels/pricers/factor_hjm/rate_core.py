import numpy as np
from numba import njit
from typing import Union, Tuple


# @njit(cache=False, fastmath=True)
def bracket(ts: np.ndarray, t: float, throw_if_not_found: bool = False) -> int:
    # if 0 in ts:
    #     raise ValueError("0 should be exluded")
    idx0 = -1
    for idx, tk in enumerate(ts):
        if t <= tk:
            idx0 = idx
            break
    if idx0 == -1 and throw_if_not_found:
        raise ValueError('t is not bracketed')
    return idx0


#@njit(cache=False, fastmath=True)
def pw_const(ts: np.ndarray,
             vs: np.ndarray,
             t: float,
             flat_extrapol: bool = False,
             shift: int = 0) -> Union[float, np.ndarray]:
    # if ts[0] < -1e-6 or ts[0] > 1e-6:
    #     raise ValueError('first abscissa must be zero')
    assert shift == 0 or shift == 1
    if ts.shape[0] - shift != vs.shape[0]:
        raise ValueError('abcsissas and ordinates must have same shape')
    value = np.nan
    idx0 = bracket(ts[shift:], t, False)
    value = vs[idx0]
    if flat_extrapol and t >= ts[-1]:
        value = vs[-1]
    return value


# @njit(cache=False, fastmath=True)
def get_default_swap_term_structure(expiry: float, tenor: float):
    freq = 1.0
    return np.arange(expiry, expiry + tenor + freq, freq)  # shift end by freq to include endpoint


@njit(cache=False, fastmath=True)
def get_futures_start_and_pmt(t0: float, lag: float,
                              libor_tenor: float = 0.25) -> Tuple[float, float]:
    start = t0 + lag
    end = start + libor_tenor
    return start, end


# numba version of curve related functions
# @njit(cache=False, fastmath=True)
def df_fast(t: Union[float, np.ndarray], ccy: str = "USD") -> float:
    # flat 0.8% zero rate for JPY and 3.5% for USD
    if ccy == "USD":
        r = 0.043
    elif ccy == "JPY":
        r = 0.008
    elif ccy == "USD_NS":
        lamda = 0.55/12
        beta1 = 0.0436
        beta2 = 0.013
        beta3 = -0.01

        t = np.maximum(t, 1e-4)
        r = beta1 + beta2*(1.0 - np.exp(-lamda * t)) / (lamda * t) + beta3*((1.0 - np.exp(-lamda * t)) / (lamda * t) - np.exp(-lamda * t))


    else:
        raise NotImplementedError

    disc_factor = np.exp(-r * t)
    return disc_factor


def bond_grad(bond_value, B_PX):
    bond_grad = np.zeros((bond_value.size, B_PX.size))
    for j, bj in enumerate(B_PX):
        bond_grad[:, j] = bond_value * bj
    return bond_grad


def swap_grad(numer0: np.ndarray,
              numer1: np.ndarray,
              denumer0: np.ndarray,
              denumer1: np.ndarray) -> np.ndarray:
    # scalar case is exceptional as we don't need to check input consistency
    if numer0.ndim == numer1.ndim == denumer0.ndim == denumer1.ndim:
        swap_grad = numer1 / denumer0 - (numer0 * denumer1) / np.power(denumer0, 2)
        return swap_grad

    assert numer0.ndim == 1 and denumer0.ndim == 1
    assert numer1.ndim == 2 and denumer1.ndim == 2 and np.all(numer1.shape == denumer1.shape)
    swap_grad = np.zeros_like(numer1)
    for j in range(numer1.shape[1]):
        swap_grad[:, j] = numer1[:, j] / denumer0 - (numer0 * denumer1[:, j]) / np.power(denumer0, 2)
    return swap_grad


#############################################################
def generate_ttms_grid(ttms: np.ndarray,
                       nb_pts: int = 11) -> np.ndarray:
    t0 = 0
    t_grid = np.array([0])
    for idx, ttm in enumerate(ttms):
        t_grid0 = np.linspace(t0, ttm, nb_pts)
        t_grid = np.concatenate((t_grid, t_grid0[1:]), axis=None)
        t0 = ttm
    return t_grid


@njit(cache=False, fastmath=True)
def to_yearfrac(d1, d2):
    return d2 - d1
#######################################################


def divide_mc(arr2d, arr1d):
    assert arr2d.ndim == 2 and arr1d.ndim == 1
    res = np.zeros_like(arr2d)
    for j, ann_derj in enumerate(arr2d.T):
        res[:, j] = ann_derj / arr1d
    return res


def prod_mc(arr2d, arr1d):
    assert arr2d.ndim == 2 and arr1d.ndim == 1
    res = np.zeros_like(arr2d)
    for j, ann_derj in enumerate(arr2d.T):
        res[:, j] = ann_derj * arr1d
    return res


# @njit(cache=False, fastmath=True)
def bond(t: float, T: float,
         x: np.ndarray, y: np.ndarray,
         B_PX: np.ndarray, B_PY: np.ndarray,
         ccy: str,
         m: int = 0) -> np.ndarray:
    """return bond value (scalar) or gradient {dB/dx_i, i = 0,..,d} (vector)"""
    assert t <= T
    if x.ndim == 2 or y.ndim == 2:
        # dim of X is # simulations x #factors
        # assume that in matrix X the row is for trials
        # column is for factors
        assert x.shape[0] == y.shape[0]
    assert m == 0 or m == 1
    # because number function cannot return different types, we wrap scalar value into numpy array
    bond_value = np.array([df_fast(T, ccy) / df_fast(t, ccy) * np.exp(-B_PX.dot(np.transpose(x)) - B_PY.dot(np.transpose(y)))])
    # make it just vector, not (1,nsims) matrix
    if x.ndim == 2 or y.ndim == 2:
        bond_value = bond_value[0, :]
    if m == 0:
        return bond_value
    elif m == 1:
        return bond_grad(bond_value, -B_PX)
    else:
        raise NotImplementedError


def swap_rate(ccy: str,
              t: float,
              ts_sw: np.ndarray) -> np.ndarray:
    denumer0 = 0
    for i in range(1, ts_sw.size):
        denumer0 = denumer0 + (ts_sw[i] - ts_sw[i - 1]) * df_fast(ts_sw[i], ccy) / df_fast(t, ccy)

    numer0 = df_fast(ts_sw[0], ccy) / df_fast(t, ccy) - df_fast(ts_sw[-1], ccy) / df_fast(t, ccy)
    value0 = numer0 / denumer0

    return value0


def libor_rate(ccy: str,
               t: float, tenor: float):
    zcb_start = df_fast(t, ccy=ccy)
    zcb_end = df_fast(t+tenor, ccy=ccy)
    libor = 1.0 / tenor * (zcb_start / zcb_end - 1.0)

    return libor


@njit(cache=False, fastmath=True)
def G(k, t, T):
    G_tT = (1.0 - np.exp(-k * (T - t))) / k
    return G_tT