"""Data-free Uniswap V2/V3 payoffs and static IL replication.

Adapted from SigmaStrats ``my_projects/lp_model/uniswap_math.py`` and
``lp_hedger.compute_impermanent_loss_hedge`` for Lipton, Lucic and Sepp (2025).
Prices are quote-token units per base token; fees and financing are excluded.
``il_payoff`` is P&L, while protection pays its negative. Funded P&L can be positive.
V3 requires ``0 < pa < p0 < pb``. ``pa=pb=None`` selects V2.
"""

import numpy as np


def lp_units(p, liquidity, pa=None, pb=None):
    """Return base and quote token units (Lemma 3.1; V2 limit when unbounded)."""
    p = np.asarray(p, dtype=float)
    if np.any(p <= 0.0):
        raise ValueError("Prices must be positive.")
    if pa is None and pb is None:
        return liquidity / np.sqrt(p), liquidity * np.sqrt(p)
    if pa is None or pb is None or not 0.0 < pa < pb:
        raise ValueError("Provide either two ordered positive bounds or neither.")
    root = np.sqrt(np.clip(p, pa, pb))
    return liquidity * (1.0 / root - 1.0 / np.sqrt(pb)), liquidity * (root - np.sqrt(pa))


def initial_position(p0, notional=1.0, pa=None, pb=None):
    """Return initial base units, quote units and liquidity (Eqs. 39-40)."""
    if p0 <= 0.0 or notional <= 0.0:
        raise ValueError("Initial price and notional must be positive.")
    if (pa is not None or pb is not None) and not (pa is not None and pb is not None
                                                and 0.0 < pa < p0 < pb):
        raise ValueError("The initial price must be strictly inside the V3 range.")
    x, y = lp_units(p0, 1.0, pa, pb)
    liquidity = notional / (p0 * x + y)
    return float(liquidity * x), float(liquidity * y), float(liquidity)


def il_payoff(p, p0, pa=None, pb=None, notional=1.0, borrowed=True):
    """Return funded/borrowed P&L from token holdings (Eqs. 23, 25, 41, 45)."""
    x0, y0, liquidity = initial_position(p0, notional, pa, pb)
    x, y = lp_units(p, liquidity, pa, pb)
    p = np.asarray(p, dtype=float)
    pnl = p * x + y - (p0 * x0 + y0)
    return pnl - x0 * (p - p0) if borrowed else pnl


def il_decomposition(p, p0, pa, pb, notional=1.0, borrowed=True):
    """Return V3 P&L using corrected Props. 3.5-3.6: L*(u0+2*u_half+u1)."""
    _, _, liquidity = initial_position(p0, notional, pa, pb)
    p = np.asarray(p, dtype=float)
    root = np.sqrt(p) * ((p >= pa) & (p <= pb))
    tails = (-np.maximum(pa - p, 0.0) / np.sqrt(pa)
             + np.maximum(p - pb, 0.0) / np.sqrt(pb)
             + 2.0 * np.sqrt(pa) * (p < pa) + 2.0 * np.sqrt(pb) * (p > pb))
    linear = (-p / np.sqrt(p0) - np.sqrt(p0) if borrowed else
              -p / np.sqrt(pb) + p0 / np.sqrt(pb) - 2.0 * np.sqrt(p0))
    return liquidity * (linear + 2.0 * root + tails)


def static_hedge(p0, pa, pb, put_strikes, call_strikes, notional=1.0, borrowed=True):
    """Return option weights for IL protection (Prop. 4.1).

    Both strictly increasing grids include p0: the final put and first call strike.
    Interior weights are slope differences. They linearly interpolate the protection
    payoff between strikes. Include strikes outside both bounds for exact tail slopes.
    Funded protection has signed weights; borrowed protection has long-option weights.
    """
    puts, calls = np.asarray(put_strikes, dtype=float), np.asarray(call_strikes, dtype=float)
    if (puts.ndim != 1 or calls.ndim != 1 or len(puts) < 2 or len(calls) < 2
            or np.any(np.diff(puts) <= 0.0) or np.any(np.diff(calls) <= 0.0)
            or puts[-1] != p0 or calls[0] != p0 or puts[0] <= 0.0):
        raise ValueError("Use increasing positive grids ending/starting at p0.")
    put_slopes = np.diff(il_payoff(puts, p0, pa, pb, notional, borrowed)) / np.diff(puts)
    call_slopes = -np.diff(il_payoff(calls, p0, pa, pb, notional, borrowed)) / np.diff(calls)
    put_weights = np.r_[0.0, -np.diff(put_slopes), put_slopes[-1]]
    call_weights = np.r_[call_slopes[0], np.diff(call_slopes), 0.0]
    return put_weights, call_weights


def hedge_payoff(p, put_strikes, call_strikes, put_weights, call_weights):
    """Return long puts, long calls and their total terminal payoff."""
    p = np.asarray(p, dtype=float)
    puts = np.sum(np.maximum(np.asarray(put_strikes)[:, None] - p, 0.0)
                  * np.asarray(put_weights)[:, None], axis=0)
    calls = np.sum(np.maximum(p - np.asarray(call_strikes)[:, None], 0.0)
                   * np.asarray(call_weights)[:, None], axis=0)
    return puts, calls, puts + calls
