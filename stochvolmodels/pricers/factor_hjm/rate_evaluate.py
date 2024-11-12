import numpy as np
from numba import njit
from stochvolmodels.pricers.factor_hjm.rate_core import to_yearfrac


@njit(cache=False, fastmath=True)
def init_mean_rev():
    return 0.025


class Discount:
    def __init__(self, currency="USD"):
        self.today = 0
        # flat 0.8% zero rate for JPY and 3.5% for USD
        if currency == "USD":
            self.r = 0.043
        elif currency == "JPY":
            self.r = 0.008
        else:
            raise NotImplementedError

    def df(self, d) -> float:
        year_frac = to_yearfrac(self.today, d)
        disc_factor = np.exp(-self.r * year_frac)
        return disc_factor


def G(t, T):
    k = init_mean_rev()
    G_tT = (1.0 - np.exp(-k * (T - t))) / k
    return G_tT


def bond(t, T, x, y,
         m: int,
         is_mc_mode: bool,
         discount: Discount = None):
    if discount is None:
        discount = Discount()
    if isinstance(t, np.ndarray) and isinstance(x, np.ndarray) and t.shape != x.shape:
        raise ValueError('size of t and x must agree')
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(t, np.ndarray) and is_mc_mode:
        raise ValueError('when x and y are vectors, only scalar t is allowed in MC simulations')
    if m < 0 or m > 4:
        raise ValueError('parameter m must be 0,1,2,3,4')
    k = init_mean_rev()
    G = (1.0 - np.exp(-k * (T - t))) / k
    bond_value = discount.df(T) / discount.df(t) * np.exp(-G * x - 0.5 * G ** 2 * y)
    return bond_value * np.power(-G, m)


def annuity(t, ts_sw: np.ndarray, x, y, m,
            discount: Discount = None,
            is_mc_mode: bool = False):
    if discount is None:
        discount = Discount()
    ann = 0
    for i in range(1, ts_sw.size):
        bond_value = bond(t, ts_sw[i], x, y, m, discount=discount, is_mc_mode=is_mc_mode)
        ann = ann + (ts_sw[i] - ts_sw[i-1]) * bond_value
    return ann


def swap_rate(t, ts_sw: np.ndarray, x, y,
              discount: Discount = None,
              is_mc_mode: bool = False):
    if discount is None:
        discount = Discount()
    if (isinstance(x, np.ndarray) and isinstance(y, float)) or (isinstance(x, float) and isinstance(y, np.ndarray)):
        raise ValueError('x and y both must be either scalar or vector')
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        if x.shape != y.shape:
            raise ValueError('when x and y are vectors, they must have same shape')

    denumer0 = 0
    denumer1 = 0
    denumer2 = 0
    denumer3 = 0
    denumer4 = 0
    for i in range(1, ts_sw.size):
        denumer0 = denumer0 + (ts_sw[i] - ts_sw[i - 1]) * bond(t, ts_sw[i], x, y, 0, discount=discount, is_mc_mode=is_mc_mode)
        denumer1 = denumer1 + (ts_sw[i] - ts_sw[i - 1]) * bond(t, ts_sw[i], x, y, 1, discount=discount, is_mc_mode=is_mc_mode)
        denumer2 = denumer2 + (ts_sw[i] - ts_sw[i - 1]) * bond(t, ts_sw[i], x, y, 2, discount=discount, is_mc_mode=is_mc_mode)
        denumer3 = denumer3 + (ts_sw[i] - ts_sw[i - 1]) * bond(t, ts_sw[i], x, y, 3, discount=discount, is_mc_mode=is_mc_mode)
        denumer4 = denumer4 + (ts_sw[i] - ts_sw[i - 1]) * bond(t, ts_sw[i], x, y, 4, discount=discount, is_mc_mode=is_mc_mode)

    numer0 = bond(t, ts_sw[0], x, y, 0, discount=discount, is_mc_mode=is_mc_mode) - bond(t, ts_sw[-1], x, y, 0, discount=discount, is_mc_mode=is_mc_mode)
    numer1 = bond(t, ts_sw[0], x, y, 1, discount=discount, is_mc_mode=is_mc_mode) - bond(t, ts_sw[-1], x, y, 1, discount=discount, is_mc_mode=is_mc_mode)
    numer2 = bond(t, ts_sw[0], x, y, 2, discount=discount, is_mc_mode=is_mc_mode) - bond(t, ts_sw[-1], x, y, 2, discount=discount, is_mc_mode=is_mc_mode)
    numer3 = bond(t, ts_sw[0], x, y, 3, discount=discount, is_mc_mode=is_mc_mode) - bond(t, ts_sw[-1], x, y, 3, discount=discount, is_mc_mode=is_mc_mode)
    numer4 = bond(t, ts_sw[0], x, y, 4, discount=discount, is_mc_mode=is_mc_mode) - bond(t, ts_sw[-1], x, y, 4, discount=discount, is_mc_mode=is_mc_mode)

    value0 = numer0 / denumer0
    value1 = numer1 / denumer0 - (numer0 * denumer1) / np.power(denumer0, 2)
    value2 = (-2 * numer1 * denumer1) / np.power(denumer0, 2) + numer2 / denumer0 + numer0 * ((2 * np.power(denumer1, 2)) / np.power(denumer0, 3) - denumer2 / np.power(denumer0, 2))
    value3 = (-3 * denumer1 * numer2) / np.power(denumer0, 2) + 3 * numer1 * ((2 * np.power(denumer1, 2)) / np.power(denumer0, 3) - denumer2 / np.power(denumer0, 2)) + numer3 / denumer0 + numer0 * ((-6 * np.power(denumer1, 3)) / np.power(denumer0, 4) + (6 * denumer1 * denumer2) / np.power(denumer0, 3) - denumer3 / np.power(denumer0, 2))
    value4 = (24 * numer0 * np.power(denumer1, 4) - 12 * denumer0 * np.power(denumer1, 2) * (2 * numer1 * denumer1 + 3 * numer0 * denumer2) + 2 * np.power(denumer0, 2) * (6 * np.power(denumer1, 2) * numer2 + 3 * numer0 * np.power(denumer2, 2) + 4 * denumer1 * (3 * numer1 * denumer2 + numer0 * denumer3)) + np.power(denumer0, 4) * numer4 - np.power(denumer0, 3) * (6 * numer2 * denumer2 + 4 * denumer1 * numer3 + 4 * numer1 * denumer3 + numer0 * denumer4)) / np.power(denumer0, 5)

    return value0, value1, value2, value3, value4


def libor_rate(t, t_start: float, t_end: float,
               x, y,
               discount: Discount = None,
               is_mc_mode: bool = False):
    if discount is None:
        discount = Discount()
    if (isinstance(x, np.ndarray) and isinstance(y, float)) or (isinstance(x, float) and isinstance(y, np.ndarray)):
        raise ValueError('x and y both must be either scalar or vector')
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        if x.shape != y.shape:
            raise ValueError('when x and y are vectors, they must have same shape')

    zcb_start = bond(t, t_start, x, y, 0, discount=discount, is_mc_mode=is_mc_mode)
    zcb_end = bond(t, t_end, x, y, 0, discount=discount, is_mc_mode=is_mc_mode)
    libor = 1.0/(t_end - t_start) * (zcb_start/zcb_end - 1.0)

    return libor