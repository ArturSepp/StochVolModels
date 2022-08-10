"""
utility functions
"""
import functools
import time
import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List
from typing import Tuple, Dict, Any, Optional


def to_flat_np_array(input_list: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(input_list).ravel()


@njit(cache=False, fastmath=True)
def set_time_grid(ttm: float, year_days: float = 360) -> Tuple[int, float, np.ndarray]:
    """
    set daily steps
    """
    nb_steps = int(np.ceil(year_days * ttm))
    grid_t = np.linspace(0.0, ttm, nb_steps + 1)
    dt = grid_t[1] - grid_t[0]
    return nb_steps, dt, grid_t


@njit(cache=False, fastmath=True)
def set_seed(value):
    """
    set seed for numba space
    """
    np.random.seed(value)


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
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
    compute histogram on defined discrete grid
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
