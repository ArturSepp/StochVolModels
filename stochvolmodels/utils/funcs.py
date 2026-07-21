"""
Numerical and timing utilities shared across the package.

Normal density and distribution built on a rational approximation to the
complementary error function so that they stay callable from numba nopython code,
plus time-grid construction for the Monte Carlo schemes, seeding, and small
container helpers.
"""
import functools
import time
import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List
from typing import Tuple, Dict, Any, Optional, Union


def to_flat_np_array(input_list: List[np.ndarray]) -> np.ndarray:
    """concatenate a list of per-maturity arrays into one flat array."""
    return np.concatenate(input_list).ravel()


@njit(cache=False, fastmath=False)
def set_time_grid(ttm: float, nb_steps_per_year: int = 360) -> Tuple[int, float, np.ndarray]:
    """
    build the simulation time grid for a maturity.

    Returns
    -------
    nb_steps : int
        ``int(ttm * nb_steps_per_year) + 1``.
    dt : float
        Uniform step, ``grid_t[1] - grid_t[0]``.
    grid_t : np.ndarray
        Grid of ``nb_steps + 1`` points spanning [0, ttm].

    Notes
    -----
    ``grid_t`` carries one more point than ``nb_steps``, so callers preallocating
    path arrays should size them from ``nb_steps``, not from
    ``nb_steps_per_year``.
    """
    nb_steps = int(ttm * nb_steps_per_year) + 1
    grid_t = np.linspace(0.0, ttm, nb_steps + 1)
    # dt = ttm / nb_steps
    dt = grid_t[1] - grid_t[0]
    return nb_steps, dt, grid_t


@njit(cache=False, fastmath=True)
def set_seed(value):
    """
    seed the random state inside numba nopython code.

    numba keeps a random state separate from the one ``np.random.seed`` sets in
    the interpreter, so this must be called to make jitted simulation
    reproducible.
    """
    np.random.seed(value)


def timer(func):
    """
    decorator printing the wall-clock runtime of the wrapped call.
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        """time the call, print the elapsed seconds and return the result."""
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def compute_histogram_data(data: np.ndarray,
                           x_grid: np.ndarray,
                           name: str = 'Histogram'
                           ) -> pd.Series:
    """
    histogram of simulated values on a fixed grid, normalized to frequencies.

    Returns a Series indexed by bin edges, ready to overlay on an analytic density.
    """
    hist_data, bin_edges = np.histogram(a=data,
                                        bins=len(x_grid)-1,
                                        range=(x_grid[0], x_grid[-1]))
    hist_data = np.append(np.array(x_grid[0]), hist_data)
    hist_data = hist_data / len(data)
    hist_data = pd.Series(hist_data, index=bin_edges, name=name)
    return hist_data


def update_kwargs(kwargs: Dict[Any, Any],
                  new_kwargs: Optional[Dict[Any, Any]]
                  ) -> Dict[Any, Any]:
    """
    update kwargs with optional kwargs dicts
    """
    local_kwargs = kwargs.copy()
    if new_kwargs is not None and not len(new_kwargs) == 0:
        local_kwargs.update(new_kwargs)
    return local_kwargs


@njit(cache=False, fastmath=True)
def erfcc(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    complementary error function by rational approximation.

    Numba-compatible replacement for ``scipy.special.erfc``, accurate to roughly
    1.2e-7 relative. Accepts a scalar or an array.
    """
    z = np.abs(x)
    t = 1. / (1. + 0.5*z)
    r = t * np.exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+
        t*(-1.13520398+t*(1.48851587+t*(-.82215223+t*0.17087277)))))))))
    fcc = np.where(np.greater(x, 0.0), r, 2.0-r)
    return fcc


@njit(cache=False, fastmath=True)
def ncdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """standard normal cumulative distribution, numba-compatible and vectorized."""
    return 1. - 0.5*erfcc(x/(np.sqrt(2.0)))


@njit(cache=False, fastmath=True)
def npdf(x: Union[float, np.ndarray], mu: float = 0.0, vol: float = 1.0) -> Union[float, np.ndarray]:
    """normal density with mean mu and standard deviation vol, numba-compatible."""
    return np.exp(-0.5*np.square((x-mu)/vol))/(vol*np.sqrt(2.0*np.pi))


def find_nearest(a: np.ndarray,
                 value: float,
                 is_sorted: bool = True,
                 is_equal_or_largest: bool = False
                 ) -> float:
    """
    return the element of ``a`` closest to ``value``.

    Parameters
    ----------
    a : np.ndarray
        Candidate values.
    value : float
        Target.
    is_sorted : bool, default True
        Use binary search. ``a`` must be sorted ascending; no check is made.
    is_equal_or_largest : bool, default False
        Return the first element at or above ``value`` rather than the nearest.
        Used for maturity lookup into a volatility backbone, where interpolating
        below the quoted tenor would extrapolate.

    Returns
    -------
    float
        The selected element of ``a``.
    """
    if is_sorted:
        idx = np.searchsorted(a, value, side="left")
        if is_equal_or_largest:  # return the equal or largest element
            return a[idx]
        else:
            if idx > 0 and (idx == len(a) or np.abs(value - a[idx - 1]) < np.abs(value - a[idx])):
                return a[idx - 1]
            else:
                return a[idx]
    else:
        a = np.asarray(a)
        idx = (np.abs(a - value)).argmin()
    return a[idx]


